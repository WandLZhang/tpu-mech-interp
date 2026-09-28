#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU gate for compare.py: its exit rule with a stub judge, and its judge on the real engine.

    python3 steering/test_compare.py

The first checks need no engine. A stub judge stands in for the model and labels replies from a
table, so the exit rule and the handling of the judge's controls run on counts this file sets.

Then the README's Run-it chain on CPU. `cpu_rig.py` serves a tiny random Gemma 4 with the real
tokenizer through the patched `sglang-jax`, `scripts/capture_activations.py` captures slot 2,
`sae/train.py` fits an SAE to it, `pick_feature.py` picks a live latent and `from_sae.py` builds
its bank. `compare.py` then runs twice, once with chat turns and once with `--raw`. A third run
steers the last layer with a direction built in `transformers`. A random tiny model judges at
random, so the engine runs prove the judge's plumbing, and the stub checks prove the rule.

Eleven checks, each with a control that has to fail:

1. A chat turn reaches the model with one BOS. The engine reports each prompt's token count, and
   it has to equal the length of the template's ids, which open on one BOS. Control: the same
   turn rendered to text and sent as text, the route compare.py used to take. The engine's
   tokenizer adds a BOS to it, so it reaches the model with two.
2. `--raw` still sends plain text, and the engine gives it one BOS, the way it treats a capture
   prompt. Its judge still sends chat turns.
3. The default log holds every payload the engine got and the metadata it sent back: the Engine's
   keywords, the template's ids, the sampling parameters, every steering request, and the judge's
   one batch, unsteered, with the controls first and each changed reply once, the template's ids
   for each question and the two label tokens it asks the log probabilities of. Control: the
   `--raw` run logs text and no ids, so the log follows what was sent.
4. The judge's plumbing returns labels on the real engine: every answer is `coherent` or
   `broken`, and it's the label the logged log probabilities pick, whatever the random model
   prefers. The exit is the one the run's own RESULT lines call for, 2 when a control got the
   wrong label, and the last line gives it. The run steers at 20 times the stream's norm so that
   both arms change every prompt and the judge gets a full batch. Control: the token the model
   itself picks first in each judge reply is never a label, so the labels come from the log
   probabilities.
5. A dead feature is refused before the engine loads. Control: the live feature's bank, under a
   name without `.npz`, gets its control row.
6. Steered at the last layer, the first new token is a function of that layer's output at the
   prompt's last position, so `transformers` predicts both arms. The bank's one direction lifts
   one prompt's runner-up token over its top token, and the seed is the first whose control
   direction changes no first token. The run takes one new token in float32. The engine has to
   change the prompts `transformers` predicts, and the exit has to follow the judge. Control: a
   hook that adds nothing changes no first token, where `transformers` predicts a change.
7. An `--engine-arg` key compare.py sets itself, such as `tp_size` or `steering_layer`, exits 2
   before the bank loads, and so does an item with no `=`. Control: `dtype=float32`, which
   compare.py leaves open, gets as far as the bank.
8. A run that fails after the bank copy, here on a model directory with no tokenizer, leaves no
   `steer-control-*` directory behind. Control: `bank_with_control` on its own, whose copy stays.
9. The last line reads the counts the summaries hold, for the counts the latent 7987 run of the
   README's chain gave on Gemma 4 26B-A4B before the layer filter: 4 of 4 prompts changed under
   both arms at three fractions. Control: the feature ahead in coherent replies at one alpha gives
   exit 0.
10. With a stub judge, the exit counts coherent changed replies. The latent 7987 run changed 4 of 4
    prompts under both arms at every fraction, with the feature coherent and the random direction
    broken at 0.1, and exits 0 at 0.1. The latent 1859 run changed 2 of 4 against 4 of 4 and exits 0
    the same way. A tie in coherent replies exits 1. Controls: the old rule, more changed prompts
    than the random direction, exits 1 on both runs.
11. With a stub judge, the controls lead one batch that holds each changed reply once and no
    unchanged one. A judge that mislabels the known-good reply or the known-broken one, answers
    outside the two labels, answers too few or raises exits 2, and the last line prints its
    answers to the controls. `label_from_logprobs` picks the label whose token the engine rates
    higher, and a tie or an unrated label gives a sentence that isn't a label. Control: the same
    replies with the controls labeled right exit 0.

Needs what `cpu_rig.py` needs: the engine's runtime imports, `git`, and network access.

If a control passes, this file fails itself.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "sae"))
import compare  # noqa: E402
import from_sae  # noqa: E402
from cpu_rig import (  # noqa: E402
    BATCH_SIZE, CPU_ENGINE, PROMPTS, TINY, TOKEN_PADDING, Rig, engine_args, results,
)
from train import TrainConfig  # noqa: E402

SLOT = 2
STEERING_LAYER = SLOT - 1
FRACTIONS = (0.1, 20.0)
MAX_NEW_TOKENS = 8
TRAIN_BATCH = 32
TRAIN_STEPS = 100
LAST_LAYER = TINY["num_hidden_layers"] - 1
# The lead in logits that each first token transformers predicts needs over its runner-up.
MARGIN = 1e-2
SEEDS = 64

failures = 0


def report(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def control(detected, text):
    global failures
    print(f"      control ({text}): {'detected' if detected else 'NOT DETECTED'}")
    if not detected:
        print("      FAIL: the control didn't fail, so this check proves nothing.")
        failures += 1


def payload_records(stderr):
    """compare.py's payload records at the default log level, one JSON object per line."""
    prefix = "INFO compare "
    return [
        json.loads(line[len(prefix):]) for line in stderr.splitlines() if line.startswith(prefix)
    ]


def last_line(stdout):
    """The last line a run printed on stdout."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def stage(rows, name):
    return [r for r in rows if r.get("stage") == name]


def expected_exit(rows):
    """The exit a run's own RESULT lines call for, worked out without compare.py's helpers."""
    controls = stage(rows, "judge_control")
    judged = stage(rows, "steer_judged")
    labels = ("coherent", "broken")
    if (len(controls) != len(compare.JUDGE_CONTROLS)
            or any(c["answer"] != c["want"] for c in controls)
            or any(j["label"] not in labels for j in judged)):
        return 2
    for s in stage(rows, "steer_summary"):
        feature = sum(j["label"] == "coherent" for j in judged
                      if j["arm"] == "feature" and round(j["alpha"], 4) == s["alpha"])
        random = sum(j["label"] == "coherent" for j in judged
                     if j["arm"] == "control" and round(j["alpha"], 4) == s["alpha"])
        if feature > random:
            return 0
    return 1


# ---------------------------------------------------------------------------------------------
# The stub judge.

GOOD = "The capital of France is Paris, a city on the Seine."
SWAP = "The capital of France is Paris, a large city on the Seine."
LOOP = "to to to to to to to to"
MARKUP = "** $ \\ ** $ \\ ** $"
SCRIPT = "의 의 의 的 的 的 и и"
# What each reply is, as a reader of the latent 7987 run judged it.
TRUTH = {GOOD: "coherent", SWAP: "coherent", LOOP: "broken", MARKUP: "broken", SCRIPT: "broken"}
TRUTH.update({c["reply"]: c["want"] for c in compare.JUDGE_CONTROLS})
ALPHAS = [10.7842, 53.9209, 107.8419]
SCALE = 107.8419
META = {"feature": 7987, "layer": 14, "mode": "static", "prompts": 4}


def stub_judge(calls, answer=None):
    """A judge that labels each reply from TRUTH, or through `answer(item, truth)` when given."""
    def judge(items):
        calls.append(items)
        return [answer(i, TRUTH[i["reply"]]) if answer else TRUTH[i["reply"]] for i in items]
    return judge


def replies(table):
    """Steered replies from `{(alpha, arm): [text or None per prompt]}`. None is an unchanged one."""
    out = []
    for alpha in ALPHAS:
        for arm in ("feature", "control"):
            for p, text in enumerate(table[(alpha, arm)]):
                out.append({"arm": arm, "alpha": alpha, "prompt": f"prompt {p}",
                            "changed": text is not None, "text": text or GOOD})
    return out


def scored(judge, steered):
    """compare.score over `steered` with `judge`. Returns (exit, last line, RESULT rows)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code, last = compare.score(judge, steered, ALPHAS, SCALE, META)
    return code, last, results(buffer.getvalue())


def old_rule(rows):
    """The rule before the judge: exit 0 when the feature changes more prompts at some alpha."""
    return 0 if any(s["changed"] > s["changed_control"]
                    for s in stage(rows, "steer_summary")) else 1


# The latent 7987 run: both arms change every prompt at every fraction. At 0.1 the feature swaps a
# word and the random direction loops; at 0.5 and 1.0 the feature writes markup and the random
# direction writes other scripts.
STAGE8 = {
    (ALPHAS[0], "feature"): [SWAP] * 4, (ALPHAS[0], "control"): [LOOP] * 4,
    (ALPHAS[1], "feature"): [MARKUP] * 4, (ALPHAS[1], "control"): [SCRIPT] * 4,
    (ALPHAS[2], "feature"): [MARKUP] * 4, (ALPHAS[2], "control"): [SCRIPT] * 4,
}
# The latent 1859 run: at 0.1 the feature changes 2 of 4 by a word and the random direction all 4.
STAGE7 = dict(STAGE8)
STAGE7[(ALPHAS[0], "feature")] = [SWAP, None, SWAP, None]


def check_exit_rule():
    """10. The exit counts coherent changed replies."""
    calls = []
    code, last, rows = scored(stub_judge(calls), replies(STAGE8))
    counts = [(s["changed"], s["changed_control"], s["coherent"], s["coherent_control"])
              for s in stage(rows, "steer_summary")]
    report(code == 0 and counts == [(4, 4, 4, 0), (4, 4, 0, 0), (4, 4, 0, 0)]
           and stage(rows, "steer_verdict")[0]["beats_control_at"] == [ALPHAS[0]]
           and last.startswith("exit 0: ") and last.endswith(f"at alpha {ALPHAS[0]}."),
           f"the latent 7987 run, changed / changed by the random direction / coherent / coherent"
           f" under the random direction per alpha {counts}: exit {code}")
    control(old_rule(rows) == 1, f"the old rule gives {old_rule(rows)} on the latent 7987 run")

    code, last, rows = scored(stub_judge([]), replies(STAGE7))
    counts = [(s["changed"], s["changed_control"], s["coherent"], s["coherent_control"])
              for s in stage(rows, "steer_summary")]
    report(code == 0 and counts[0] == (2, 4, 2, 0),
           f"the latent 1859 run, 2 of 4 against 4 of 4 at 0.1, all the feature's coherent:"
           f" {counts[0]},"
           f" exit {code}")
    control(old_rule(rows) == 1, f"the old rule gives {old_rule(rows)} on the latent 1859 run")

    tie = dict(STAGE8)
    tie[(ALPHAS[0], "control")] = [SWAP, SWAP, SWAP, SWAP]
    code, last, rows = scored(stub_judge([]), replies(tie))
    counts = [(s["coherent"], s["coherent_control"]) for s in stage(rows, "steer_summary")]
    report(code == 1 and "Exit 0 needs more coherent changed replies" in last,
           f"4 coherent against 4 at 0.1 and none after, {counts}: exit {code}")
    ahead = dict(tie)
    ahead[(ALPHAS[1], "feature")] = [SWAP, MARKUP, MARKUP, MARKUP]
    code, _, rows = scored(stub_judge([]), replies(ahead))
    control(code == 0, f"one more coherent reply under the feature at 0.5 gives exit {code}")


def check_judge_controls():
    """11. The controls lead one batch, and a judge that fails them exits 2."""
    calls = []
    steered = replies(STAGE7)
    code, last, rows = scored(stub_judge(calls), steered)
    changed = [(r["prompt"], r["text"]) for r in steered if r["changed"]]
    sent = [(i["prompt"], i["reply"]) for i in calls[0][len(compare.JUDGE_CONTROLS):]] if calls else []
    leads = [(i["prompt"], i["reply"]) for i in calls[0][:len(compare.JUDGE_CONTROLS)]] if calls else []
    report(len(calls) == 1
           and leads == [(c["prompt"], c["reply"]) for c in compare.JUDGE_CONTROLS]
           and sent == changed and len(sent) == 22,
           f"one judge batch: the {len(leads)} controls, then the {len(sent)} changed replies in"
           f" order and none of the 2 unchanged ones")
    judged = stage(rows, "steer_judged")
    report([(j["prompt"], j["text"], j["label"]) for j in judged]
           == [(p, t, TRUTH[t]) for p, t in changed],
           f"each of the {len(judged)} changed replies prints with its label")
    control(code == 0, f"with the controls labeled right the run exits {code}")

    def flip(want):
        def answer(item, truth):
            return ("broken" if truth == "coherent" else "coherent") if item["about"] == \
                f"control, known {want}" else truth
        return answer

    cases = {
        "mislabels the known-good reply": (flip("coherent"), "known coherent got 'broken'"),
        "mislabels the known-broken reply": (flip("broken"), "known broken got 'coherent'"),
        "answers outside the two labels":
            (lambda item, truth: "Coherent." if item["about"].startswith("feature") else truth,
             "answered 'Coherent."),
    }
    for name, (answer, want) in cases.items():
        code, last, rows = scored(stub_judge([], answer), steered)
        report(code == 2 and want in last and last.startswith("exit 2: the judge ")
               and stage(rows, "steer_verdict")[0]["exit"] == 2
               and len(stage(rows, "judge_control")) == 2,
               f"a judge that {name} exits {code}: {last}")

    def short(items):
        return ["coherent", "broken"]

    def raises(items):
        raise RuntimeError("the engine stopped")

    for name, judge, want in (("answers two of 24", short, "gave 2 answers to 24 requests"),
                              ("raises", raises, "raised RuntimeError: the engine stopped")):
        code, last, _ = scored(judge, steered)
        report(code == 2 and want in last, f"a judge that {name} exits {code}: {last}")

    # output_token_ids_logprobs as the engine sends it: per position, (logprob, id, text).
    ids = [11, 22]

    def meta(*pairs):
        return {"output_token_ids_logprobs": [[[lp, i, None] for lp, i in pairs]]}

    picks = {
        "coherent rated higher": (meta((-0.5, 11), (-2.0, 22)), "coherent"),
        "broken rated higher": (meta((-3.0, 11), (-0.1, 22)), "broken"),
    }
    for name, (info, want) in picks.items():
        got = compare.label_from_logprobs(info, ids)
        report(got == want, f"label_from_logprobs, {name}: {got!r}")
    for name, info in (("a tie", meta((-1.0, 11), (-1.0, 22))),
                       ("broken unrated", meta((-1.0, 11))),
                       ("no ratings at all", {})):
        got = compare.label_from_logprobs(info, ids)
        report(got not in compare.LABELS, f"label_from_logprobs, {name}: {got!r}, no label")


# ---------------------------------------------------------------------------------------------
# The engine runs.

def build_bank(rig, root):
    """Capture, train, pick and convert, as the README's steps 2 to 4 do. Returns the pieces."""
    prompts = os.path.join(root, "prompts.txt")
    with open(prompts, "w") as fp:
        fp.write("\n".join(PROMPTS) + "\n")
    caps = os.path.join(root, "caps")
    proc = rig.run(
        "scripts/capture_activations.py",
        "--model-path", rig.model, "--prompts", prompts, "--out", caps, "--layers", SLOT,
        "--dtype", "float32", "--tp-size", 1, "--batch-size", BATCH_SIZE,
        "--token-padding", TOKEN_PADDING, "--max-new-tokens", 3, *engine_args(),
    )
    if proc.returncode != 0:
        return None
    manifest = os.path.join(caps, "manifest.json")
    with open(manifest) as fp:
        tokens = json.load(fp)["tokens"]
    # The command line keeps TrainConfig's calibration batches, and every batch is a fresh one.
    batches = TRAIN_STEPS + TrainConfig.calibration_batches
    passes = math.ceil(batches * TRAIN_BATCH / tokens) + 1
    sae = os.path.join(root, "sae_l2.npz")
    proc = rig.run(
        "sae/train.py", "--manifest", manifest, "--layer", SLOT, "--expansion-factor", 16,
        "--k", 4, "--steps", TRAIN_STEPS, "--warmup-steps", 10, "--batch-size", TRAIN_BATCH,
        "--learning-rate", 1e-3, "--passes", passes, "--shuffle-bytes", 1 << 20, "--out", sae,
    )
    if proc.returncode != 0:
        return None
    proc = rig.run("steering/pick_feature.py", "--sae", sae, "--manifest", manifest)
    last = last_line(proc.stdout)
    if proc.returncode != 0 or not last.startswith("FEATURE="):
        return None
    feature = int(last.partition("=")[2])
    bank = os.path.join(root, "steer_l2.npz")
    proc = rig.run("steering/from_sae.py", "--sae", sae, "--feature", feature, "--out", bank)
    if proc.returncode != 0 or f"--steering-layer {STEERING_LAYER}" not in proc.stdout:
        return None
    return sae, feature, bank


def push_bank(rig, root, tokenizer, chat_ids):
    """A bank whose one direction changes one prompt's first new token at the last layer.

    Steered at the last layer, the first new token is a function of that layer's output at the
    prompt's last position, through the final norm, the LM head and the softcap. The direction is
    the gradient that lifts the runner-up's logit over the top token's, on the prompt the shortest
    push flips, and alpha is twice the length where it flips. The seed is the first whose control
    direction, drawn by compare.py's own `bank_with_control`, changes no first token. Every
    predicted top token leads its runner-up by MARGIN and decodes to clean text.

    Returns `(bank, alpha, seed, first tokens off, first tokens on)`, or None.
    """
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    with open(os.path.join(rig.model, "config.json")) as fp:
        text = json.load(fp)["text_config"]
    config = Gemma4TextConfig(**{k: v for k, v in text.items() if k != "model_type"})
    model = Gemma4ForCausalLM.from_pretrained(rig.model, config=config, dtype=torch.float32).eval()

    streams, tokens = [], []

    def grab(module, args, out):
        streams.append(out[0, -1].detach().clone())
        tokens.append(out[0].detach().clone())

    hook = model.model.layers[LAST_LAYER].register_forward_hook(grab)
    with torch.no_grad():
        for ids in chat_ids:
            model(input_ids=torch.tensor([ids]))
    hook.remove()
    # The RMS token norm at the steered site, the scale from_sae.py stores.
    scale = float(torch.cat(tokens).pow(2).sum(-1).mean().sqrt())
    cap = config.final_logit_softcapping
    special = set(tokenizer.all_special_ids)

    def logits(h):
        z = model.lm_head(model.model.norm(h))
        return torch.tanh(z / cap) * cap if cap else z

    def first_tokens(push):
        """Each prompt's first new token under `push`, and the smallest lead over a runner-up."""
        with torch.no_grad():
            tops = [torch.topk(logits(h + push), 2) for h in streams]
        ids = [int(t.indices[0]) for t in tops]
        return ids, min(float(t.values[0] - t.values[1]) for t in tops)

    def clean(ids):
        return all(t not in special and "\ufffd" not in tokenizer.decode([t]) for t in ids)

    off, lead = first_tokens(torch.zeros_like(streams[0]))
    if lead < MARGIN or not clean(off):
        return None
    candidates = []
    for p, h in enumerate(streams):
        x = h.clone().requires_grad_(True)
        z = logits(x)
        values = z.detach()
        runner_up = int(torch.topk(values, 2).indices[1])
        (z[runner_up] - z[off[p]]).backward()
        grad = x.grad.detach()
        # The first-order estimate of the length that ties the two logits. Prompts are tried
        # shortest first.
        tie = float(values[off[p]] - values[runner_up]) / float(grad.norm())
        candidates.append((tie, p, grad / grad.norm()))
    for tie, p, direction in sorted(candidates, key=lambda c: c[0]):
        def flips(length):
            return first_tokens(length * direction)[0][p] != off[p]

        hi = 2 * tie
        for _ in range(30):
            if flips(hi):
                break
            hi *= 2
        else:
            continue
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if flips(mid) else (mid, hi)
        alpha = 2 * hi
        on, lead = first_tokens(alpha * direction)
        changed = [(a, b) for a, b in zip(off, on) if a != b]
        if lead < MARGIN or not clean(on) or any(
            tokenizer.decode([a]) == tokenizer.decode([b]) for a, b in changed
        ):
            continue
        vector = direction.numpy()[None, :].astype(np.float32)
        bank = from_sae.save_bank(
            os.path.join(root, "push_bank.npz"),
            from_sae.SteeringBank(
                features=np.asarray([0], np.int32),
                vectors=vector,
                probes=vector,
                thresholds=np.zeros(1, np.float32),
                scales=np.asarray([scale], np.float32),
            ),
            meta={"steering_layer": LAST_LAYER},
        )
        for seed in range(SEEDS):
            copy, _, _ = compare.bank_with_control(bank, 0, seed)
            with np.load(copy) as data:
                row = torch.from_numpy(data["vectors"][-1])
            shutil.rmtree(os.path.dirname(copy), ignore_errors=True)
            kept, lead = first_tokens(alpha * row)
            if kept == off and lead >= MARGIN:
                return bank, alpha, seed, off, on
    return None


def engine_arg_exit(root, item):
    """compare.py's exit code for one `--engine-arg` and a bank that isn't there.

    A refused item exits 2 from argparse. An accepted one gets as far as the bank, where the
    missing file raises, and that returns "parsed".
    """
    argv = [
        "--model-path", os.path.join(root, "absent"), "--bank", os.path.join(root, "absent.npz"),
        "--steering-layer", STEERING_LAYER, "--feature", 0, "--engine-arg", item,
    ]
    try:
        compare.main([str(a) for a in argv])
    except SystemExit as exc:
        return exc.code
    except FileNotFoundError:
        return "parsed"
    return "ran"


def left_behind(root, run):
    """Call `run` with tempfile's directory pointed at a fresh one.

    Returns the bank copies left in it. Other names don't count: `torch`, which `transformers`
    imports, makes a cache directory there on its own.
    """
    scratch = tempfile.mkdtemp(dir=root)
    saved = tempfile.tempdir
    tempfile.tempdir = scratch
    try:
        run()
    except BaseException as exc:  # noqa: BLE001 - the check reads what the run left behind
        print(f"      the run raised {type(exc).__name__}: {exc}")
    finally:
        tempfile.tempdir = saved
    return sorted(name for name in os.listdir(scratch) if name.startswith("steer-control-"))


def check_cleanup(root):
    """8. A run that fails after the bank copy removes the copy's directory."""
    vector = np.eye(1, 8, dtype=np.float32)
    bank = from_sae.save_bank(
        os.path.join(root, "tiny_bank.npz"),
        from_sae.SteeringBank(
            features=np.asarray([0], np.int32), vectors=vector, probes=vector,
            thresholds=np.zeros(1, np.float32), scales=np.ones(1, np.float32),
        ),
    )
    no_tokenizer = os.path.join(root, "no-tokenizer")
    os.makedirs(no_tokenizer, exist_ok=True)
    argv = [
        "--model-path", no_tokenizer, "--bank", bank, "--steering-layer", STEERING_LAYER,
        "--feature", 0,
    ]
    left = left_behind(root, lambda: compare.main([str(a) for a in argv]))
    report(not left, f"a run that fails on a model with no tokenizer leaves {left or 'nothing'}")
    kept = left_behind(root, lambda: compare.bank_with_control(bank, 0, 0))
    control(bool(kept), f"bank_with_control on its own leaves {kept}")


def check_verdict_line():
    """9. The last line gives the exit code and reads the counts off the summaries."""
    def summary(alpha, fraction, changed, changed_control, coherent, coherent_control):
        return {"alpha": alpha, "fraction_of_norm": fraction, "changed": changed,
                "changed_control": changed_control, "coherent": coherent,
                "coherent_control": coherent_control}

    # The counts the latent 7987 run of the README's chain gave on Gemma 4 26B-A4B before the
    # layer filter, scale 107.8419, with no coherent reply ahead of the random direction's.
    tied = [summary(10.7842, 0.1, 4, 4, 0, 0), summary(53.9209, 0.5, 4, 4, 0, 0),
            summary(107.8419, 1.0, 4, 4, 0, 0)]
    line = compare.verdict_line(tied, 4, 1)
    want = ("exit 1: at alpha 10.7842, 53.9209 and 107.8419, 0.1, 0.5 and 1.0 of the norm, the"
            " feature changed 4, 4 and 4 of 4 prompts and the random direction 4, 4 and 4. The"
            " judge called 0, 0 and 0 of the feature's changed replies coherent and 0, 0 and 0 of"
            " the random direction's.")
    report(line.startswith(want) and compare.feature_beats_control(tied) == []
           and compare.exit_code(tied, None) == 1,
           f"4 of 4 under both arms at three fractions: {line}")
    ahead = [summary(3.5, None, 1, 4, 1, 0)]
    line = compare.verdict_line(ahead, 4, 0)
    control(line.startswith("exit 0:") and "at alpha 3.5." in line
            and compare.exit_code(ahead, None) == 0,
            f"a feature ahead in coherent replies at one alpha: {line}")


def main() -> int:
    print("the exit rule, with a stub judge")
    check_exit_rule()
    print("\nthe judge's controls, with a stub judge")
    check_judge_controls()

    with tempfile.TemporaryDirectory() as root:
        print("\n--engine-arg keys compare.py sets itself")
        for item in ("tp_size=2", "steering_layer=3", "steering_bank=x.npz",
                     "enable_steering=false", "model_path=x", "batch_size=2", "token_padding=64",
                     "no_equals_sign"):
            code = engine_arg_exit(root, item)
            report(code == 2, f"--engine-arg {item} exits {code} before the bank loads")
        code = engine_arg_exit(root, "dtype=float32")
        control(code == "parsed", f"dtype=float32, which compare.py leaves open, gives {code!r}")

        print("\nthe bank copy on a failed run")
        check_cleanup(root)
        print("\nthe last line")
        check_verdict_line()

        rig = Rig(root)
        print("\nthe Run-it chain through step 4's bank")
        built = build_bank(rig, root)
        report(built is not None, f"capture, SAE, pick and bank at capture slot {SLOT}")
        if built is None:
            return 1
        sae, feature, bank = built

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(rig.model)
        bos = tokenizer.bos_token_id
        prompts = list(compare.DEFAULT_PROMPTS)

        def chat(texts):
            """Each text as one user turn: the rendered template, then the token ids.

            The template's text already holds its BOS, so encoding it with no special tokens is
            the one-BOS sequence. Written without compare.py's own helper.
            """
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
                )
                for t in texts
            ]
            return rendered, [tokenizer(r, add_special_tokens=False)["input_ids"] for r in rendered]

        rendered, chat_ids = chat(prompts)
        raw_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        one_bos = all(ids[0] == bos and ids[1] != bos for ids in chat_ids)
        common = [
            "--model-path", rig.model, "--bank", bank, "--steering-layer", STEERING_LAYER,
            "--feature", feature, "--tp-size", 1, "--batch-size", BATCH_SIZE,
            "--token-padding", TOKEN_PADDING, "--max-new-tokens", MAX_NEW_TOKENS,
            "--fractions", *FRACTIONS, *engine_args(),
        ]

        print("\ncompare.py with chat turns")
        run = rig.run("steering/compare.py", *common)
        rows = results(run.stdout)
        setup = stage(rows, "steer_setup")
        off = stage(rows, "steer_off")
        on = stage(rows, "steer_on")
        summaries = stage(rows, "steer_summary")
        verdict = stage(rows, "steer_verdict")
        finished = bool(verdict) and len(off) == len(prompts) and len(summaries) == len(FRACTIONS)
        report(finished, f"the run finished, exit {run.returncode}")
        if not finished:
            return 1

        counts = [r["prompt_tokens"] for r in off]
        template_counts = [len(ids) for ids in chat_ids]
        report(
            setup[0]["input"] == "token_ids" and one_bos and counts == template_counts,
            f"each chat turn prefills the template's ids, one BOS then the turn: engine counts"
            f" {counts}, template ids {template_counts}",
        )

        records = payload_records(run.stderr)
        engines = [r for r in records if r["event"] == "engine"]
        sent = [r for r in records if r["event"] == "generate"]
        replies_logged = [r for r in records if r["event"] == "reply"]
        judged_sent = [r for r in records if r["event"] == "judge"]
        answers = [r for r in records if r["event"] == "judge_reply"]
        want_engine = {
            "model_path": rig.model, "enable_steering": True, "steering_layer": STEERING_LAYER,
            "tp_size": 1, "max_running_requests": BATCH_SIZE,
            "precompile_token_paddings": [TOKEN_PADDING], **CPU_ENGINE,
        }
        engine_ok = (
            len(engines) == 1
            and all(engines[0].get(k) == v for k, v in want_engine.items())
            and os.path.basename(engines[0].get("steering_bank", "")) == os.path.basename(bank)
        )
        logged = json.dumps(engines[0]) if engines else None
        report(engine_ok, f"the log holds the Engine's keywords once: {logged}")
        control_id = setup[0]["control"]
        arms = (("feature", feature), ("control", control_id))
        want_arms = [("off", None)] + [pair for _ in FRACTIONS for pair in arms]

        def steering_logged(record, fid):
            if fid is None:
                return "steering" not in record
            request = {"feature": fid, "alpha": record["alpha"], "mode": "static"}
            return record["steering"] == [request] * len(prompts)

        payloads_ok = len(sent) == len(want_arms) and all(
            r["input_ids"] == chat_ids
            and "prompt" not in r
            and r["sampling_params"] == {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0}
            and r["arm"] == arm
            and steering_logged(r, fid)
            for r, (arm, fid) in zip(sent, want_arms)
        )
        meta_ok = len(replies_logged) == len(want_arms) * len(prompts) and all(
            r["meta_info"]["prompt_tokens"] == len(chat_ids[i % len(prompts)])
            for i, r in enumerate(replies_logged)
        )
        report(
            payloads_ok and meta_ok,
            f"the default log holds all {len(sent)} steered payloads, ids, sampling parameters"
            f" and steering requests, and the metadata of all {len(replies_logged)} replies",
        )

        changed = [(r["prompt"], r["text"]) for r in on if r["changed"]]
        want_items = [(c["prompt"], c["reply"]) for c in compare.JUDGE_CONTROLS] + changed
        # Each label's first token, as a model turn opens with it. Written without compare.py's
        # own helper.
        label_ids = [tokenizer(label, add_special_tokens=False)["input_ids"][0]
                     for label in ("coherent", "broken")]
        judge_ok = len(judged_sent) == 1
        if judge_ok:
            items = judged_sent[0]["items"]
            questions = [compare.JUDGE_TEMPLATE.format(prompt=i["prompt"], reply=i["reply"])
                         for i in items]
            judge_ok = (
                [(i["prompt"], i["reply"]) for i in items] == want_items
                and judged_sent[0]["sampling_params"] == {"max_new_tokens": 1, "temperature": 0.0}
                and judged_sent[0]["return_logprob"] is True
                and judged_sent[0]["token_ids_logprob"] == label_ids
                and judged_sent[0].get("input_ids") == chat(questions)[1]
                and "steering" not in judged_sent[0]
                and len(answers) == len(items)
                and all(a["meta_info"].get("prompt_tokens") == len(ids)
                        for a, ids in zip(answers, chat(questions)[1]))
            )
        report(
            judge_ok,
            f"the log holds one judge batch, unsteered, that asks for the log probabilities of"
            f" tokens {label_ids}: the 2 controls and the {len(changed)} changed replies as"
            f" chat-template ids, and {len(answers)} replies with their metadata",
        )

        def rated_label(record):
            """The label the logged log probabilities pick, or None."""
            positions = record["meta_info"].get("output_token_ids_logprobs") or [[]]
            rated = {int(e[1]): float(e[0]) for e in positions[0]}
            if not all(i in rated for i in label_ids):
                return None
            coherent, broken = (rated[i] for i in label_ids)
            return "coherent" if coherent > broken else "broken" if broken > coherent else None

        picked = [rated_label(a) for a in answers]
        labels = [a["answer"] for a in answers]
        report(
            bool(labels) and all(t in ("coherent", "broken") for t in labels)
            and picked == labels
            and [j["label"] for j in stage(rows, "steer_judged")] == labels[2:]
            and [c["answer"] for c in stage(rows, "judge_control")] == labels[:2],
            f"the judge's plumbing returns labels on the real engine, the ones the logged log"
            f" probabilities pick: {json.dumps(labels)}",
        )
        greedy = [a["text"] for a in answers]
        control(
            bool(greedy) and not any(t in ("coherent", "broken") for t in greedy),
            f"the token the model itself picks first in each judge reply is never a label:"
            f" {json.dumps(greedy)}",
        )
        want = expected_exit(rows)
        final = last_line(run.stdout)
        report(
            run.returncode == want and verdict[0]["exit"] == want
            and final.startswith(f"exit {want}: "),
            f"the exit is the one the run's RESULT lines call for: exit {run.returncode}, expected"
            f" {want}; last line: {final}",
        )

        print("\ncompare.py with --raw, two prompts plain and two rendered chat turns")
        raw_prompts = prompts[:2] + rendered[:2]
        prompt_args = [a for p in raw_prompts for a in ("--prompt", p)]
        raw = rig.run("steering/compare.py", *common, *prompt_args, "--raw")
        rows = results(raw.stdout)
        raw_off = stage(rows, "steer_off")
        raw_setup = stage(rows, "steer_setup")
        raw_counts = [r["prompt_tokens"] for r in raw_off]
        plain_want = [len(ids) + 1 for ids in raw_ids[:2]]
        report(
            bool(raw_setup) and raw_setup[0]["input"] == "text" and raw_counts[:2] == plain_want,
            f"--raw sends text and the engine adds one BOS: engine counts {raw_counts[:2]},"
            f" text ids plus one {plain_want}",
        )
        doubled = [n + 1 for n in template_counts[:2]]
        control(
            raw_counts[2:] == doubled,
            f"a rendered chat turn sent as text prefills {raw_counts[2:]} tokens, one more than the"
            f" template's {template_counts[:2]}, because the engine adds a second BOS",
        )
        raw_records = payload_records(raw.stderr)
        raw_sent = [r for r in raw_records if r["event"] == "generate"]
        control(
            bool(raw_sent)
            and all(r.get("prompt") == raw_prompts and "input_ids" not in r for r in raw_sent),
            "the --raw run logs the text it sent and no ids",
        )
        raw_judge = [r for r in raw_records if r["event"] == "judge"]
        report(
            len(raw_judge) == 1 and "input_ids" in raw_judge[0] and "prompt" not in raw_judge[0],
            "the --raw run's judge still sends chat-template ids",
        )
        report(
            raw.returncode == expected_exit(rows),
            f"--raw exits {raw.returncode}, which its own RESULT lines call for",
        )

        print("\na dead feature")
        params, theta, _ = from_sae.load_sae(sae)
        dead = np.flatnonzero(~np.isfinite(np.asarray(theta)))
        report(dead.size > 0, f"the SAE has {dead.size} dead latent(s) to test with")
        if dead.size:
            # from_sae.py refuses to write this bank, so it comes from the library, the way a bank
            # built by hand or by an older from_sae.py would.
            dead_feature = int(dead[0])
            dead_bank = from_sae.save_bank(
                os.path.join(root, "dead_bank.npz"),
                from_sae.build_bank(params, theta, [dead_feature]),
            )
            proc = rig.run(
                "steering/compare.py", "--model-path", rig.model, "--bank", dead_bank,
                "--steering-layer", STEERING_LAYER, "--feature", dead_feature, "--tp-size", 1,
                *engine_args(),
            )
            report(
                proc.returncode != 0 and "is dead" in proc.stderr and not results(proc.stdout),
                f"--feature {dead_feature}, threshold inf, exits {proc.returncode} before the"
                f" engine loads",
            )
        no_suffix = os.path.join(root, "steer_l2")
        shutil.copy(bank, no_suffix)
        path, _, _ = compare.bank_with_control(no_suffix, feature, seed=0)
        exists = os.path.exists(path)
        rows_held = []
        if exists:
            with np.load(path) as data:
                rows_held = [int(f) for f in data["features"]]
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        control(
            exists and len(rows_held) == 2 and rows_held[0] == feature,
            f"the live feature's bank, named without .npz, gets its control row at {path}",
        )

        print("\ncompare.py at the last layer, with a direction built to change one first token")
        pushed = push_bank(rig, root, tokenizer, chat_ids)
        report(
            pushed is not None,
            "transformers finds a direction that changes one first token and a control seed that"
            " changes none",
        )
        if pushed is not None:
            push, alpha, seed, off_ids, on_ids = pushed
            proc = rig.run(
                "steering/compare.py", "--model-path", rig.model, "--bank", push,
                "--steering-layer", LAST_LAYER, "--feature", 0, "--alphas", alpha, "--seed", seed,
                "--tp-size", 1, "--batch-size", BATCH_SIZE, "--token-padding", TOKEN_PADDING,
                "--max-new-tokens", 1, *engine_args(dtype="float32"),
            )
            rows = results(proc.stdout)
            got = [r["text"] for r in stage(rows, "steer_off")]
            predicted = [tokenizer.decode([t]) for t in off_ids]
            report(
                got == predicted,
                f"the engine's first tokens are the ones transformers predicts: {json.dumps(got)}",
            )
            on = stage(rows, "steer_on")
            flips = [r["changed"] for r in on if r["arm"] == "feature"]
            control_flips = [r["changed"] for r in on if r["arm"] == "control"]
            want_flips = [a != b for a, b in zip(off_ids, on_ids)]
            want = expected_exit(rows)
            final = last_line(proc.stdout)
            report(
                flips == want_flips
                and control_flips == [False] * len(prompts)
                and proc.returncode == want
                and final.startswith(f"exit {want}: "),
                f"alpha {alpha:.4f}, seed {seed}: the feature changes {sum(flips)} prompt(s) where"
                f" transformers predicts {sum(want_flips)}, the control {sum(control_flips)}"
                f" -> exit {proc.returncode}, which the judge's labels call for; last line: {final}",
            )
            control(
                any(want_flips),
                f"a hook that adds nothing changes no first token, where transformers predicts"
                f" {sum(want_flips)} change(s)",
            )

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
