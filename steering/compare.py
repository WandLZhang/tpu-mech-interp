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

"""Generate from the same prompts with steering off, then on, beside a random direction as control.

    python3 steering/compare.py --model-path SNAPSHOT_DIR --bank steer_l20.npz \
        --steering-layer 19 --feature 40977 --tp-size 8

One engine load serves every arm, because steering is a field on each request, not a server
setting beyond the bank and the layer. Prints each prompt's text off and on, then each changed
reply with the label the judge gave it, then one count line per alpha. The last line gives the
exit code and the counts behind it.

The hook adds `alpha * v` for a unit vector `v`, so alpha is a length in activation units. The
bank stores the SAE's input scale, the RMS token norm at the capture slot, as each feature's
scale. `--fractions` sets alpha as a share of that norm, 0.1, 0.5 and 1.0 by default. On Gemma 4
26B-A4B at slot 15 the README's chain measured the norm at 107.49 on 2026-09-25, so fraction 1.0
was alpha 107.49. `--alphas` takes lengths directly.

Greedy decoding turns on near-ties, so a small push anywhere can change a few tokens. The control
arm steers a random unit direction, seeded by `--seed`, at the same alphas, with the feature's
probe, threshold and scale.

A count of changed prompts can't tell a coherent word swap from broken text. In a run of the
README's chain on Gemma 4 26B-A4B before the layer filter, latent 7987 changed 4 of 4 prompts at
fractions 0.1, 0.5 and 1.0, and so did the random direction. At 0.1 the feature swapped a word or
two and kept the text coherent, while the random direction fell into "to to to". So the served
model judges every changed reply. It runs unsteered and gets the prompt and the reply as one user
turn that asks for one word, `coherent` or `broken`. The engine returns the log probability of
each label's first token at the first position of the answer, and the label with the higher one
is the answer. No text gets parsed. The judge's requests go in one batch, and two controls of
known label lead it: `JUDGE_CONTROLS` holds a coherent reply and a broken one. The exit code reads:

- 2 when the judge mislabels a control or gives no label, for example when the engine rates
  neither label's token. The judge's answers to the controls print, and the counts don't set the
  exit.
- 0 when the feature has more coherent changed replies than the random direction at some alpha.
- 1 otherwise.

argparse also exits 2, on a bad command line, before the engine loads.

The judge doesn't use the engine's grammar path. At eb061d8 the grammar cache hands one llguidance
matcher to every request in a batch that asks for the same grammar. On the CPU rig a batch of 18
answers under `coherent|broken` came back as fragments such as "co", "herent" and "br". Upstream
merged a fix in sgl-project/sglang-jax#1710 on 2026-09-28, after eb061d8.

`--steering-layer` is one less than the capture slot the SAE was trained on: capture reads the
stream entering a block and the hook writes the stream leaving the one you name. `from_sae.py`
prints the right value.

A model with a chat template gets each prompt as one user turn, because an instruction-tuned model
continues raw text badly: Gemma 4 26B-A4B-it continues "The capital of France is" with " DO laite-"
on the CPU reference too. The turn goes to the engine as the token ids the template gives. Gemma's
template writes its own BOS and the engine's tokenizer adds one to any text it encodes, so the
rendered text would reach the model with two, where the capture the SAE came from had one. `--raw`
sends the prompts as plain text, which the engine encodes the way it encodes a capture prompt.

A dead feature, one whose threshold is `inf`, is refused before the engine loads. Conditional
steering never fires it, and static steering adds its decoder row to every token it steers, a
direction the SAE never writes at inference.

The log on stderr holds every payload the engine gets: the Engine's keywords, then each request's
token ids or text, sampling parameters and steering requests, the judge's requests with the text
each one asks about, and the text and metadata of every reply. `--log-level warning` turns it off.

The bank copy with the control row lives in a temporary directory, which goes whether the run
finishes or fails.

Engine() re-imports this module in its subprocesses, so everything sits behind `__main__`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))

DEFAULT_PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered",
    "The best way to cook rice is",
    "Climate policy over the next decade will",
]

# Keys compare.py sets itself, and the flag that sets each. `--engine-arg` refuses them, because
# the Engine would get the keyword twice. `dtype` stays open, since nothing here sets it.
RESERVED_ENGINE_ARGS = {
    "model_path": "use --model-path",
    "enable_steering": "compare.py always turns it on",
    "steering_bank": "use --bank",
    "steering_layer": "use --steering-layer",
    "tp_size": "use --tp-size",
    "batch_size": "use --batch-size",
    "token_padding": "use --token-padding",
}

# The judge's two labels. The answer is the one whose first token the model rates higher.
LABELS = ("coherent", "broken")
# One position is all the answer needs: the engine rates both labels' first tokens there.
JUDGE_MAX_NEW_TOKENS = 1
# A model with no chat template reads the question as text and answers after this.
JUDGE_TEXT_TAIL = "\nAnswer:"
JUDGE_TEMPLATE = (
    "A language model wrote the reply below to the prompt below.\n\n"
    "Prompt: {prompt}\n\n"
    "Reply: {reply}\n\n"
    "Is the reply coherent text, or is it broken, for example repeated words, stray symbols or "
    "tokens from other scripts? The reply stops at a length limit, so a sentence cut off at the "
    "end can still be coherent. Answer with one word: coherent or broken."
)
# Two replies of known label that lead every judge batch. A judge that mislabels either one makes
# the run exit 2. The broken one is the failure the random direction showed on Gemma 4 26B-A4B.
JUDGE_CONTROLS = (
    {
        "want": "coherent",
        "prompt": "The capital of France is",
        "reply": "Paris. It is the largest city in France, and it sits on the river Seine.",
    },
    {
        "want": "broken",
        "prompt": "The best way to cook rice is",
        "reply": "to to to to to to to to to to to to to to to to to to to to to to to to to to",
    },
)
JUDGE_ERROR = 2

# The engine adds a root handler and sets the root level when it loads, so this logger keeps its
# own handler and doesn't propagate.
log = logging.getLogger("compare")


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s compare %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(level.upper())
    log.propagate = False


def bank_with_control(path: str, feature: int, seed: int) -> tuple[str, int, float]:
    """Copy the bank with one extra row: a random unit direction beside `feature`.

    The row takes the feature's probe, threshold and scale, so conditional mode fires on the same
    tokens. Its id sits one past the SAE's width, which no real latent uses. Returns the copy's
    path, the control's id and the feature's scale. A feature the bank doesn't hold, or holds
    with a threshold that isn't finite, exits.
    """
    with np.load(path, allow_pickle=False) as data:
        bank = {k: data[k] for k in data.files}
    features = [int(f) for f in bank["features"]]
    if feature not in features:
        raise SystemExit(f"feature {feature} isn't in {path}; it holds {features}")
    row = features.index(feature)
    threshold = float(bank["thresholds"][row])
    if not np.isfinite(threshold):
        raise SystemExit(
            f"feature {feature} is dead in {path}: its threshold is {threshold}. Conditional "
            f"steering never fires it, and static steering adds its decoder row to every token it "
            f"steers, a direction the SAE never writes at inference. Pick a live latent with "
            f"steering/pick_feature.py."
        )
    meta = json.loads(str(bank["meta"])) if "meta" in bank else {}
    config = meta.get("sae_config") or {}
    width = config.get("d_sae") or (
        config["d_model"] * config["expansion_factor"]
        if "d_model" in config and "expansion_factor" in config
        else max(features) + 1
    )
    control = int(width)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=bank["vectors"].shape[1])
    direction /= np.linalg.norm(direction)
    bank["features"] = np.append(bank["features"], np.int32(control)).astype(np.int32)
    bank["vectors"] = np.vstack([bank["vectors"], direction[None, :]]).astype(np.float32)
    bank["probes"] = np.vstack([bank["probes"], bank["probes"][row][None, :]]).astype(np.float32)
    for key in ("thresholds", "scales"):
        bank[key] = np.append(bank[key], bank[key][row]).astype(np.float32)
    # `np.savez` adds `.npz` to a name that lacks it, so name the copy with it.
    name = os.path.basename(path)
    name = name if name.endswith(".npz") else name + ".npz"
    out = os.path.join(tempfile.mkdtemp(prefix="steer-control-"), name)
    np.savez(out, **bank)
    return out, control, float(bank["scales"][row])


def prompt_inputs(tokenizer, prompts: list, raw: bool) -> tuple[list, bool]:
    """What the engine gets for each prompt. Returns `(inputs, as_ids)`.

    With a chat template, and without `raw`, each prompt becomes one user turn as the token ids
    `apply_chat_template` gives, the ids the engine's own chat endpoint sends. They skip the
    engine's tokenizer, so the template's BOS is the only one. Otherwise each prompt goes as
    text.

    Raises:
      ValueError: the template itself writes more than one leading BOS.
    """
    if raw or not getattr(tokenizer, "chat_template", None):
        return list(prompts), False
    bos = tokenizer.bos_token_id
    inputs = []
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = [int(i) for i in ids]
        leading = 0
        while bos is not None and leading < len(ids) and ids[leading] == bos:
            leading += 1
        if leading > 1:
            raise ValueError(f"the chat template writes {leading} BOS tokens ahead of {prompt!r}")
        inputs.append(ids)
    return inputs, True


def judge_inputs(tokenizer, items: list) -> tuple[list, bool]:
    """The judge's question about each `{prompt, reply}` item, as `prompt_inputs` sends a prompt.

    With a chat template the question goes as one user turn, whatever `--raw` says, because
    `--raw` is about the steered prompts. Without one the question goes as text that ends on
    `JUDGE_TEXT_TAIL`.
    """
    texts = [JUDGE_TEMPLATE.format(prompt=i["prompt"], reply=i["reply"]) for i in items]
    if not getattr(tokenizer, "chat_template", None):
        texts = [t + JUDGE_TEXT_TAIL for t in texts]
    return prompt_inputs(tokenizer, texts, raw=False)


def label_token_ids(tokenizer, as_ids: bool) -> list:
    """The first token of each label, as the judge's answer would open with it.

    In a chat turn the answer opens the model's turn, so the label starts bare. After
    `JUDGE_TEXT_TAIL` it follows a space.

    Raises:
      ValueError: two labels open on the same token, so their log probabilities can't differ.
    """
    prefix = "" if as_ids else " "
    ids = [int(tokenizer(prefix + label, add_special_tokens=False)["input_ids"][0])
           for label in LABELS]
    if len(set(ids)) != len(ids):
        raise ValueError(f"the labels {list(LABELS)} open on the same token: {ids}")
    return ids


def label_from_logprobs(meta_info: dict, ids: list) -> str:
    """The label whose first token has the higher log probability at the answer's first position.

    Reads `output_token_ids_logprobs`, which the engine fills from `token_ids_logprob`. Returns a
    sentence that isn't a label when the engine rated a label's token nowhere or rated both the
    same, so the caller sees the judge failed.
    """
    positions = meta_info.get("output_token_ids_logprobs") or []
    rated = {int(entry[1]): float(entry[0]) for entry in (positions[0] if positions else [])}
    missing = [label for label, i in zip(LABELS, ids) if i not in rated]
    if missing:
        return f"no log probability for {_spoken(missing)}"
    scores = [rated[i] for i in ids]
    best = max(scores)
    if scores.count(best) > 1:
        return f"a tie at log probability {best}"
    return LABELS[scores.index(best)]


def engine_judge(engine, tokenizer):
    """A judge that asks the served model, unsteered, and reads its log probabilities.

    The returned callable takes a list of items, each a dict with `prompt` and `reply` and any
    keys that say what the item is, and returns each answer: a label, or a sentence that says why
    there's none. Every payload and every reply goes to the log.
    """
    sampling = {"max_new_tokens": JUDGE_MAX_NEW_TOKENS, "temperature": 0.0}

    def judge(items: list) -> list:
        inputs, as_ids = judge_inputs(tokenizer, items)
        ids = label_token_ids(tokenizer, as_ids)
        payload = {
            "input_ids" if as_ids else "prompt": inputs,
            "sampling_params": sampling,
            "return_logprob": True,
            "token_ids_logprob": ids,
        }
        log.info(json.dumps({
            "event": "judge",
            "labels": {label: i for label, i in zip(LABELS, ids)},
            "items": items,
            **payload,
        }))
        outputs = engine.generate(**payload)
        answers = []
        for item, output in zip(items, outputs):
            answer = label_from_logprobs(output["meta_info"], ids)
            log.info(
                json.dumps(
                    {
                        "event": "judge_reply",
                        **item,
                        "answer": answer,
                        "text": output["text"],
                        "meta_info": output["meta_info"],
                    },
                    default=str,
                )
            )
            answers.append(answer)
        return answers

    return judge


def judge_replies(judge, changed: list) -> tuple[list, list, str | None]:
    """Ask `judge` about the controls and every changed reply, in one batch.

    `changed` holds dicts with `arm`, `alpha`, `prompt` and `text`. Returns the label for each
    changed reply, each control with the judge's `answer`, and a sentence that says what went
    wrong, or None. An answer that isn't a label, a control the judge mislabels, a count of
    answers that doesn't match and a judge that raises each give that sentence.
    """
    items = [
        {"about": f"control, known {c['want']}", "prompt": c["prompt"], "reply": c["reply"]}
        for c in JUDGE_CONTROLS
    ] + [
        {"about": f"{r['arm']} at alpha {r['alpha']}", "prompt": r["prompt"], "reply": r["text"]}
        for r in changed
    ]
    try:
        answers = list(judge(items))
    except Exception as exc:  # noqa: BLE001 - the run reports it and exits 2
        answers, problem = [], f"raised {type(exc).__name__}: {exc}."
    else:
        problem = None
    controls = [
        dict(c, answer=answers[n] if n < len(answers) else None)
        for n, c in enumerate(JUDGE_CONTROLS)
    ]
    labels = answers[len(JUDGE_CONTROLS):]
    if problem is None and len(answers) != len(items):
        problem = f"gave {len(answers)} answers to {len(items)} requests."
    if problem is None:
        stray = [a for a in answers if a not in LABELS]
        if stray:
            problem = f"answered {_spoken([repr(a) for a in stray])}, outside {list(LABELS)}."
    if problem is None:
        wrong = [c for c in controls if c["answer"] != c["want"]]
        if wrong:
            problem = "mislabeled " + _spoken(
                [f"the known {c['want']} reply as {c['answer']!r}" for c in wrong]
            ) + "."
    return labels, controls, problem


def summarize(replies: list, labels: list | None, alphas: list, scale: float, meta: dict) -> list:
    """One summary per alpha: prompts changed and changed replies judged coherent, per arm.

    `replies` holds every steered reply in order, each with `alpha`, `arm` and `changed`. `labels`
    holds one label per changed reply in the same order, or None when the judge failed, which
    leaves the coherent counts at None.
    """
    labeled = iter(labels or [])
    counts = {}
    for r in replies:
        key = (r["alpha"], r["arm"])
        changed, coherent = counts.get(key, (0, 0))
        if r["changed"]:
            label = next(labeled, None)
            counts[key] = (changed + 1, coherent + (label == "coherent"))
        else:
            counts[key] = (changed, coherent)
    summaries = []
    for alpha in alphas:
        feature = counts.get((alpha, "feature"), (0, 0))
        control = counts.get((alpha, "control"), (0, 0))
        summaries.append(
            {
                "stage": "steer_summary",
                **meta,
                "alpha": round(alpha, 4),
                "fraction_of_norm": round(alpha / scale, 4) if scale else None,
                "changed": feature[0],
                "changed_control": control[0],
                "coherent": feature[1] if labels is not None else None,
                "coherent_control": control[1] if labels is not None else None,
            }
        )
    return summaries


def feature_beats_control(summaries: list) -> list:
    """The alphas at which the feature had more coherent changed replies than the control."""
    return [
        s["alpha"]
        for s in summaries
        if s["coherent"] is not None and s["coherent"] > s["coherent_control"]
    ]


def exit_code(summaries: list, problem: str | None) -> int:
    """2 when the judge failed, 0 when the feature beats the control at some alpha, else 1."""
    if problem is not None:
        return JUDGE_ERROR
    return 0 if feature_beats_control(summaries) else 1


def _spoken(items: list) -> str:
    """`a`, `a and b`, `a, b and c`."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def verdict_line(summaries: list, prompts: int, code: int, problem: str | None = None,
                 controls: list | None = None) -> str:
    """The last line compare.py prints: the exit code and what set it."""
    if problem is not None:
        answers = _spoken(
            [f"known {c['want']} got {c['answer']!r}" for c in controls]
        ) if controls else "none"
        return (
            f"exit {code}: the judge {problem} Its answers to the controls: {answers}. The counts"
            f" don't set the exit when the judge fails."
        )
    alphas = _spoken([s["alpha"] for s in summaries])
    fractions = [s["fraction_of_norm"] for s in summaries]
    where = f"at alpha {alphas}"
    if all(f is not None for f in fractions):
        where += f", {_spoken(fractions)} of the norm"
    line = (
        f"exit {code}: {where}, the feature changed {_spoken([s['changed'] for s in summaries])}"
        f" of {prompts} prompts and the random direction"
        f" {_spoken([s['changed_control'] for s in summaries])}. The judge called"
        f" {_spoken([s['coherent'] for s in summaries])} of the feature's changed replies coherent"
        f" and {_spoken([s['coherent_control'] for s in summaries])} of the random direction's."
    )
    beaten = feature_beats_control(summaries)
    if beaten:
        return line + f" The feature is ahead of the random direction at alpha {_spoken(beaten)}."
    return line + (
        " Exit 0 needs more coherent changed replies under the feature than under the random"
        " direction at one alpha."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-path", required=True, help="the local snapshot directory")
    ap.add_argument("--bank", required=True, help="the .npz from_sae.py wrote")
    ap.add_argument("--steering-layer", type=int, required=True)
    ap.add_argument("--feature", type=int, required=True, help="a live latent; see pick_feature.py")
    ap.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="alpha as a share of the stream's RMS norm, which the bank stores as the scale",
    )
    ap.add_argument("--alphas", type=float, nargs="+", help="alpha as a length, in place of --fractions")
    ap.add_argument("--seed", type=int, default=0, help="seeds the control's random direction")
    ap.add_argument("--mode", default="static", choices=["static", "conditional"])
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to four built-ins")
    ap.add_argument("--raw", action="store_true", help="plain text, even when the model has a chat template")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--token-padding", type=int, default=1024)
    ap.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Engine keyword, repeatable; VALUE reads as JSON, True, False, None or a string",
    )
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning"],
        help="info and debug log every payload the engine gets and every reply; warning doesn't",
    )
    args = ap.parse_args(argv)
    from capture_activations import parse_engine_args

    try:
        extra = parse_engine_args(args.engine_arg, reserved=RESERVED_ENGINE_ARGS)
    except ValueError as exc:
        ap.error(str(exc))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    setup_logging(args.log_level)

    bank_path, control, scale = bank_with_control(args.bank, args.feature, args.seed)
    try:
        code = steer(args, extra, bank_path, control, scale)
    finally:
        shutil.rmtree(os.path.dirname(bank_path), ignore_errors=True)
    # `os._exit` skips interpreter cleanup, which is why the bank copy goes in the `finally` above.
    os._exit(code)


def steer(args, extra: dict, bank_path: str, control: int, scale: float) -> int:
    """Everything after the bank copy: the engine, every arm, the judge. Returns the exit."""
    from capture_activations import ENGINE_LOAD_NOTE, engine_settings

    alphas = args.alphas or [f * scale for f in args.fractions]

    prompts = args.prompt or DEFAULT_PROMPTS
    # The tokenizer loads before sgl_jax is imported, which registers its own config classes.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    inputs, as_ids = prompt_inputs(tokenizer, prompts, args.raw)
    print(
        "RESULT "
        + json.dumps(
            {
                "stage": "steer_setup",
                "chat_template": as_ids,
                "input": "token_ids" if as_ids else "text",
                "prompts": len(prompts),
                "feature": args.feature,
                "control": control,
                "scale": scale,
                "alphas": [round(a, 4) for a in alphas],
            }
        )
    )

    from sgl_jax.srt.entrypoints.engine import Engine

    settings = engine_settings(
        batch_size=args.batch_size, token_padding=args.token_padding, tp_size=args.tp_size, **extra
    )
    engine_kwargs = {
        "model_path": args.model_path,
        "enable_steering": True,
        "steering_bank": bank_path,
        "steering_layer": args.steering_layer,
        **settings,
    }
    log.info(json.dumps({"event": "engine", **engine_kwargs}, default=str))
    print(ENGINE_LOAD_NOTE, flush=True)
    engine = Engine(**engine_kwargs)
    try:
        code, last = run_arms(
            engine, args, prompts, inputs, as_ids, alphas, control, scale, tokenizer
        )
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    # After the engine stops, so nothing it prints lands below the line.
    print(last)
    return code


def run_arms(engine, args, prompts, inputs, as_ids, alphas, control, scale,
             tokenizer) -> tuple[int, str]:
    """Generate unsteered, then the feature and the control at each alpha, then judge.

    Returns the exit code and the last line to print.
    """
    sampling = {"max_new_tokens": args.max_new_tokens, "temperature": 0.0}

    def generate(arm, alpha, steering=None):
        payload = {"input_ids" if as_ids else "prompt": inputs, "sampling_params": sampling}
        if steering is not None:
            payload["steering"] = steering
        log.info(json.dumps({"event": "generate", "arm": arm, "alpha": alpha, **payload}))
        outputs = engine.generate(**payload)
        for prompt, output in zip(prompts, outputs):
            log.info(
                json.dumps(
                    {
                        "event": "reply",
                        "arm": arm,
                        "alpha": alpha,
                        "prompt": prompt,
                        "text": output["text"],
                        "meta_info": output["meta_info"],
                    },
                    default=str,
                )
            )
        return outputs

    off = generate("off", 0.0)
    for p, o in zip(prompts, off):
        print(
            "RESULT "
            + json.dumps(
                {
                    "stage": "steer_off",
                    "prompt": p,
                    "prompt_tokens": o["meta_info"].get("prompt_tokens"),
                    "text": o["text"],
                }
            )
        )
    replies = []
    for alpha in alphas:
        for arm, feature in (("feature", args.feature), ("control", control)):
            request = {"feature": feature, "alpha": alpha, "mode": args.mode}
            on = generate(arm, alpha, [request] * len(prompts))
            for p, a, b in zip(prompts, off, on):
                reply = {"arm": arm, "alpha": alpha, "prompt": p,
                         "changed": a["text"] != b["text"], "text": b["text"]}
                replies.append(reply)
                print("RESULT " + json.dumps({"stage": "steer_on", **reply}))
    meta = {"feature": args.feature, "layer": args.steering_layer, "mode": args.mode,
            "prompts": len(prompts)}
    return score(engine_judge(engine, tokenizer), replies, alphas, scale, meta)


def score(judge, replies: list, alphas: list, scale: float, meta: dict) -> tuple[int, str]:
    """Judge the changed replies, print what the judge said and the counts, and set the exit.

    `replies` holds every steered reply, each a dict with `arm`, `alpha`, `prompt`, `changed` and
    `text`. `meta` holds `feature`, `layer`, `mode` and `prompts`, which every summary carries.
    Prints the judge's answer to each control, each changed reply with its label, one count line
    per alpha and the verdict. Returns the exit code and the last line to print.
    """
    changed = [r for r in replies if r["changed"]]
    labels, controls, problem = judge_replies(judge, changed)
    for c in controls:
        print("RESULT " + json.dumps({"stage": "judge_control", **c}))
    for n, r in enumerate(changed):
        label = labels[n] if n < len(labels) else None
        print(
            "RESULT "
            + json.dumps(
                {"stage": "steer_judged", "arm": r["arm"], "alpha": r["alpha"],
                 "prompt": r["prompt"], "label": label, "text": r["text"]}
            )
        )
    summaries = summarize(replies, None if problem else labels, alphas, scale, meta)
    for s in summaries:
        print("RESULT " + json.dumps(s))
    code = exit_code(summaries, problem)
    verdict = {
        "stage": "steer_verdict",
        "feature": meta["feature"],
        "prompts": meta["prompts"],
        "changed": [s["changed"] for s in summaries],
        "changed_control": [s["changed_control"] for s in summaries],
        "coherent": [s["coherent"] for s in summaries],
        "coherent_control": [s["coherent_control"] for s in summaries],
        "judge_problem": problem,
        "beats_control_at": feature_beats_control(summaries),
        "exit": code,
    }
    print("RESULT " + json.dumps(verdict))
    return code, verdict_line(summaries, meta["prompts"], code, problem, controls)


if __name__ == "__main__":
    sys.exit(main())
