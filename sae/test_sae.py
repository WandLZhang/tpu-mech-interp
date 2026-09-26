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

"""Correctness gate for the BatchTopK sparse autoencoder and its JumpReLU conversion.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 sae/test_sae.py

The data is synthetic with a known answer: a fixed dictionary of unit-norm atoms, a few positive
coefficients per token, and a little noise. Check 3 measures the learned dictionary against that
generating dictionary, so every claim here has a number attached to it.

Thirteen checks, each with a negative control that has to fail:

0. The identities the module documents hold. Sparse decode equals dense decode, the `shard_map`
   decode on the 8-device mesh equals the one-device decode, folding the pre-encoder bias leaves
   pre-activations alone, the projected decoder gradient is orthogonal to every latent vector,
   and the BatchTopK window is empty. Control: each identity re-run with one sign or one term
   wrong, `b_dec` added once per shard for the `shard_map` decode.
1. Training lowers the fraction of variance unexplained. Control: the untrained SAE.
2. JumpReLU reproduces the BatchTopK codes on held-out tokens. Controls: the same thresholds
   permuted across latents, rescaled by 0.75, and with dead latents opened up to zero.
3. The learned dictionary recovers the generating atoms. Control: a random dictionary.
4. Sharding the dictionary over the mesh, with the `shard_map` decode, doesn't change the
   answer. Control: the shard contents rotated by one device, evaluated on the mesh.
5. `train.py` records the capture slot a checkpoint was trained on, and only that. On a 3-D shard
   read through `--activations`, `--layer` is an axis position and never lands in the record.
   Controls: `--capture-layer` on the same shards, the same slot read through a manifest, and a
   2-D shard, all of which record the slot.
6. `step_flops` counts what the compiled step runs, within 5% of XLA's own count. Control: the
   count that treats the sparse decoder as a dense matmul.
7. A batch too wide for one BatchTopK selection fails at trace time with a message that names
   the batch to use, on both kernels and in `train.py` before it reads a shard. Control: the
   widest batch that fits traces.
8. A bfloat16 forward pass trains float32 parameters. After two real updates the parameters, the
   gradients Adam reads and its state are float32, and the forward ran at bfloat16. Controls: a
   step that returns bfloat16 parameters, and a step that stores the Adam state in bfloat16.
9. On the mesh, the compiled training step all-reduces `[batch, d_model]` partial sums and no
   `[k * batch, d_model]` block of gathered rows. Control: the same step built without the mesh.
10. Every matmul asks for `HIGHEST` precision: the training step, the threshold fit, the JumpReLU
    forward and the bias fold. Control: the encoder matmul written without a precision.
11. `train` stops on a NaN or an infinity in the scale sample, in a training batch and in a
    calibration batch. Control: the same stream without them trains to finite parameters.
12. `train.py` counts the 16 batches the scale fit reads. At `--steps 4` it refuses a capture of
    68 batches, which the steps plus 64 calibration batches would cover, and one of 80 calibrates
    on all 64. Control: the 80-batch capture trains.

If a control passes, this file fails itself.
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import tempfile

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sae as sae_lib  # noqa: E402
import train as train_lib  # noqa: E402
from sae import LATENT_AXIS, SAEConfig  # noqa: E402
from train import TrainConfig, train  # noqa: E402

D_MODEL = 32
EXPANSION = 4
K_TRUE = 3
ATOMS = 64
TOKENS = 16_384


def make_activations(seed, tokens=TOKENS, d_model=D_MODEL, atoms=ATOMS, k_true=K_TRUE):
    """Tokens built from a known sparse code over a known dictionary.

    Coefficients are positive, since the SAE only represents positive magnitudes. Atom norms
    vary, so the encoder has to learn scale and not only direction.
    """
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(atoms, d_model)).astype(np.float32)
    dictionary /= np.linalg.norm(dictionary, axis=-1, keepdims=True)
    dictionary *= (0.5 + rng.random((atoms, 1))).astype(np.float32)

    codes = np.zeros((tokens, atoms), dtype=np.float32)
    rows = np.repeat(np.arange(tokens), k_true)
    cols = np.concatenate(
        [rng.choice(atoms, size=k_true, replace=False) for _ in range(tokens)]
    )
    codes[rows, cols] = 1.0 + rng.exponential(size=tokens * k_true).astype(np.float32)

    x = codes @ dictionary
    x += 0.01 * rng.normal(size=x.shape).astype(np.float32)
    return x.astype(np.float32), dictionary


def batch_stream(pool, batch_size, seed):
    """Endless random batches drawn from a fixed pool."""
    rng = np.random.default_rng(seed)
    while True:
        yield pool[rng.integers(0, pool.shape[0], size=batch_size)]


def fvu(params, cfg, x, theta=None):
    """Fraction of variance unexplained on one batch.

    Passing `theta` runs the JumpReLU path, otherwise BatchTopK.
    """
    if theta is None:
        recon, _, _ = sae_lib.forward_batchtopk(params, cfg, x, subtract_pre_bias=False)
    else:
        recon, _ = sae_lib.forward_jumprelu(params, x, theta)
    err = jnp.sum((x - recon) ** 2)
    var = jnp.sum((x - jnp.mean(x, axis=0)) ** 2)
    return float(err / jnp.maximum(var, 1e-8))


def code_agreement(params, cfg, x, theta):
    """How closely the JumpReLU active set matches the BatchTopK one.

    Returns `(jaccard, relative reconstruction difference, l0_batchtopk, l0_jumprelu)`.
    """
    _, values, indices = sae_lib.forward_batchtopk(params, cfg, x, subtract_pre_bias=False)
    dense_bt = sae_lib.scatter_dense(values, indices, x.shape[0], cfg.d_sae)
    recon_bt = sae_lib.decode_dense(params, dense_bt)
    recon_jr, dense_jr = sae_lib.forward_jumprelu(params, x, theta)

    on_bt = dense_bt > 0
    on_jr = dense_jr > 0
    union = jnp.sum(on_bt | on_jr)
    jaccard = float(jnp.sum(on_bt & on_jr) / jnp.maximum(union, 1))
    diff = float(
        jnp.linalg.norm(recon_jr - recon_bt) / jnp.maximum(jnp.linalg.norm(recon_bt), 1e-8)
    )
    return jaccard, diff, float(jnp.mean(jnp.sum(on_bt, axis=-1))), float(
        jnp.mean(jnp.sum(on_jr, axis=-1))
    )


def recovery(learned_w_dec, truth):
    """Best cosine from each generating atom to some learned latent vector.

    Direction only. `unscale_params` leaves the learned rows carrying activation units, and the
    generating atoms carry their own norms, so comparing lengths would measure the input scale.
    """
    learned = np.asarray(learned_w_dec, dtype=np.float64)
    learned /= np.maximum(np.linalg.norm(learned, axis=-1, keepdims=True), 1e-12)
    atoms = np.asarray(truth, dtype=np.float64)
    atoms /= np.maximum(np.linalg.norm(atoms, axis=-1, keepdims=True), 1e-12)
    return (atoms @ learned.T).max(axis=1)


def check_identities(params, cfg, x, mesh):
    """Check 0. One allclose per documented identity, each with a broken twin."""
    failures = 0

    def report(name, got, want, control_got, tol=1e-4):
        nonlocal failures
        ok = got < tol
        detected = control_got > tol
        status = "PASS" if ok and detected else "FAIL"
        print(
            f"  [{status}] {name:<24} err={got:.2e}  control={control_got:.2e}"
            f" -> {'detected' if detected else 'NOT DETECTED'}"
        )
        failures += not (ok and detected)

    batch = x.shape[0]
    pre = sae_lib.encode_pre(params, x, subtract_pre_bias=False)
    count = sae_lib.batch_topk_count(cfg.k, batch, cfg.d_sae)
    values, indices = sae_lib.batch_topk(pre, count, cfg.recall_target)
    dense = sae_lib.scatter_dense(values, indices, batch, cfg.d_sae)

    sparse = sae_lib.decode_sparse(params, values, indices, batch)
    report(
        "sparse == dense decode",
        float(jnp.max(jnp.abs(sparse - sae_lib.decode_dense(params, dense)))),
        0.0,
        float(jnp.max(jnp.abs(sparse - sae_lib.decode_dense(params, 2.0 * dense)))),
    )

    # The `shard_map` decode on the mesh against the one-device decode. Every shard holds a copy
    # of `b_dec`, so the twin adds it once per shard, `(shards - 1) * b_dec` high.
    shards = sae_lib.latent_shards(mesh)
    on_mesh = sae_lib.shard_params(params, mesh)
    picked = jax.device_put((values, indices), NamedSharding(mesh, P()))
    decode = jax.jit(sae_lib.decode_sparse, static_argnames=("batch", "add_bias", "mesh"))
    local = np.asarray(decode(on_mesh, *picked, batch=batch, mesh=mesh))
    partial = np.asarray(decode(on_mesh, *picked, batch=batch, add_bias=False, mesh=mesh))
    per_shard_bias = partial + shards * np.asarray(params.b_dec)
    report(
        f"shard_map decode on {shards}",
        float(np.max(np.abs(local - np.asarray(sparse)))),
        0.0,
        float(np.max(np.abs(per_shard_bias - np.asarray(sparse)))),
    )

    biased = params._replace(b_dec=jax.random.normal(jax.random.key(17), params.b_dec.shape))
    folded = sae_lib.encode_pre(sae_lib.fold_pre_encoder_bias(biased), x)
    report(
        "fold == subtract",
        float(jnp.max(jnp.abs(folded - sae_lib.encode_pre(biased, x, subtract_pre_bias=True)))),
        0.0,
        float(
            jnp.max(
                jnp.abs(
                    sae_lib.encode_pre(
                        biased._replace(b_enc=biased.b_enc + biased.b_dec @ biased.w_enc), x
                    )
                    - sae_lib.encode_pre(biased, x, subtract_pre_bias=True)
                )
            )
        ),
    )

    grads = sae_lib.SAEParams(
        *[jax.random.normal(jax.random.key(i), a.shape) for i, a in enumerate(params)]
    )
    rows = params.w_dec / jnp.linalg.norm(params.w_dec, axis=-1, keepdims=True)
    projected = sae_lib.project_decoder_grad(grads, params).w_dec
    report(
        "decoder grad orthogonal",
        float(jnp.max(jnp.abs(jnp.sum(projected * rows, axis=-1)))),
        0.0,
        float(jnp.max(jnp.abs(jnp.sum(grads.w_dec * rows, axis=-1)))),
        tol=1e-3,
    )

    # The window the threshold fit interpolates inside has to be empty of occurrences of the
    # latent it belongs to. That's what makes an exact top-k a threshold rule.
    stats = sae_lib.threshold_stats(pre, cfg)
    kept_min = np.asarray(stats.kept_min)
    cutoff = float(stats.cutoff)
    column = np.asarray(pre)
    live = np.isfinite(kept_min)
    inside = int(
        sum(
            np.sum((column[:, i] >= cutoff) & (column[:, i] < kept_min[i]))
            for i in np.flatnonzero(live)
        )
    )
    # Move every threshold one notch up and the window stops being empty.
    loosened = np.asarray(
        [np.max(column[:, i]) if live[i] else np.inf for i in range(cfg.d_sae)]
    )
    control = int(
        sum(
            np.sum((column[:, i] >= cutoff) & (column[:, i] < loosened[i]))
            for i in np.flatnonzero(live)
        )
    )
    ok = inside == 0 and control > 0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'topk window empty':<24} inside={inside}"
        f"  control={control} -> {'detected' if control > 0 else 'NOT DETECTED'}"
    )
    failures += not ok
    return failures


def dtype_names(tree):
    """The sorted dtype names of every array leaf in a pytree."""
    leaves = jax.tree.leaves(tree)
    return sorted({jnp.dtype(leaf.dtype).name for leaf in leaves if hasattr(leaf, "dtype")})


def check_bf16_training(x):
    """8. A bfloat16 forward pass trains float32 parameters.

    Adam's second moment follows the parameter dtype, and bfloat16 stalls it, so the master
    weights and the optimizer state have to stay float32 while the forward runs lower. Two real
    updates at `dtype=bfloat16` read the parameters, the gradients Adam gets and its state. Two
    broken steps have to fail the same gate: one returns bfloat16 parameters, one stores the Adam
    state in bfloat16.
    """
    failures = 0
    half = SAEConfig(d_model=D_MODEL, expansion_factor=EXPANSION, k=K_TRUE, dtype=jnp.bfloat16)
    full = SAEConfig(d_model=D_MODEL, expansion_factor=EXPANSION, k=K_TRUE)
    optimizer = train_lib.build_optimizer(TrainConfig())
    batch = jnp.asarray(x[:256])

    def run(step_fn, cfg):
        params = sae_lib.init_params(jax.random.key(0), cfg)
        state = optimizer.init(params)
        losses = []
        for _ in range(2):
            params, state, metrics = step_fn(params, state, batch)
            losses.append(float(metrics["loss"]))
        return params, state, losses

    def gate(params, state):
        state_ok = set(dtype_names(state)) <= {"float32", "int32"}
        return dtype_names(params) == ["float32"] and state_ok

    step = train_lib.make_step(half, optimizer)
    params, state, losses = run(step, half)
    _, _, full_losses = run(train_lib.make_step(full, optimizer), full)
    grads = jax.grad(lambda p: sae_lib.reconstruction_loss(p, half, batch)[0])(
        sae_lib.init_params(jax.random.key(0), half)
    )
    recon, _, _ = sae_lib.forward_batchtopk(params, half, batch)
    ok = (
        gate(params, state)
        and dtype_names(grads) == ["float32"]
        and recon.dtype == jnp.bfloat16
        and losses[0] != full_losses[0]
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'bf16 forward, f32 state':<24} after 2 updates: params"
        f" {dtype_names(params)}, grads {dtype_names(grads)}, Adam state {dtype_names(state)},"
        f" forward {jnp.dtype(recon.dtype).name}, first loss {losses[0]:.6f} against"
        f" {full_losses[0]:.6f} at float32"
    )
    failures += not ok

    def returns_bf16_params(p, s, xb):
        p, s, metrics = step(p, s, xb)
        return sae_lib.cast_params(p, jnp.bfloat16), s, metrics

    def stores_bf16_state(p, s, xb):
        p, s, metrics = step(p, s, xb)
        narrow = jax.tree.map(
            lambda a: a.astype(jnp.bfloat16) if jnp.issubdtype(a.dtype, jnp.floating) else a, s
        )
        return p, narrow, metrics

    for name, broken in (
        ("a step that returns bfloat16 parameters", returns_bf16_params),
        ("a step that stores the Adam state in bfloat16", stores_bf16_state),
    ):
        p, s, _ = run(broken, half)
        detected = not gate(p, s)
        print(
            f"      control ({name}): params {dtype_names(p)}, Adam state {dtype_names(s)}"
            f" -> {'detected' if detected else 'NOT DETECTED'}"
        )
        failures += not detected
    return failures


HERE = os.path.dirname(os.path.abspath(__file__))


def run_train(args):
    """`train.py` the way the command line runs it, on one CPU device."""
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "train.py"), *[str(a) for a in args]],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    print(f"    $ train.py {' '.join(str(a) for a in args)}  -> exit {proc.returncode}")
    for name, text in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        for line in text.splitlines():
            print(f"      {name} | {line}")
    return proc


def checkpoint(path):
    """The config and the arrays `train.save` wrote."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files if k != "config"}
        return json.loads(str(data["config"])), arrays


def write_capture(out_dir, rows, layers):
    """Rows as one shard plus a manifest, written by the capture script's own writer."""
    sys.path.insert(0, os.path.join(HERE, os.pardir, "scripts"))
    import capture_activations as capture  # noqa: E402

    os.makedirs(out_dir, exist_ok=True)
    shard = os.path.join(out_dir, "shard-00000.npy")
    writer = capture.ShardWriter(shard, rows.shape[1:], np.float32)
    writer.append(rows)
    record = writer.close()
    capture.write_manifest(
        out_dir,
        {
            "model": "synthetic",
            "layers": list(layers),
            "layer_axis": rows.ndim == 3,
            "dtype": "float32",
            "d_model": int(rows.shape[-1]),
            "tokens": record["tokens"],
            "shards": [record],
        },
    )
    return os.path.join(out_dir, capture.MANIFEST_NAME)


def check_capture_slot(root):
    """5. The checkpoint records the capture slot it was trained on, and never an axis position.

    Three slots, 10, 20 and 30, each drawn from its own dictionary, go into one 3-D shard. Every
    run trains two steps on it and reads `capture_layer` back out of the checkpoint.
    """
    failures = 0
    slots = (10, 20, 30)
    rows = np.stack(
        [make_activations(seed=40 + s, tokens=4_096)[0] for s in slots], axis=1
    )  # [tokens, 3, d_model]
    manifest = write_capture(os.path.join(root, "caps3"), rows, slots)
    shards3 = os.path.join(root, "caps3", "shard-*.npy")
    flat_manifest = write_capture(os.path.join(root, "caps2"), rows[:, 1, :], (20,))
    shards2 = os.path.join(root, "caps2", "shard-*.npy")
    common = [
        "--expansion-factor", EXPANSION, "--k", K_TRUE, "--steps", 2, "--warmup-steps", 1,
        "--batch-size", 32, "--shuffle-bytes", 1 << 20,
    ]

    def trained(name, *args):
        out = os.path.join(root, f"{name}.npz")
        proc = run_train([*args, *common, "--out", out])
        if proc.returncode != 0 or not os.path.exists(out):
            return proc, None, None
        config, arrays = checkpoint(out)
        return proc, config.get("capture_layer"), arrays

    # Axis position 1 of this capture holds slot 20. Recording 1 would send from_sae.py to
    # --steering-layer 0, a block the SAE never read.
    proc, slot, _ = trained(
        "glob_axis", "--activations", shards3, "--layer", 1, "--d-model", D_MODEL
    )
    ok = proc.returncode == 0 and slot is None and "records no capture slot" in proc.stdout
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'glob axis not recorded':<24}"
        f" --activations on a 3-D shard with --layer 1 records capture_layer={slot}"
    )
    failures += not ok

    proc, slot_named, glob_arrays = trained(
        "glob_named", "--activations", shards3, "--layer", 1, "--capture-layer", 20,
        "--d-model", D_MODEL,
    )
    _, slot_manifest, manifest_arrays = trained("manifest", "--manifest", manifest, "--layer", 20)
    same = (
        glob_arrays is not None
        and manifest_arrays is not None
        and all(np.array_equal(glob_arrays[k], manifest_arrays[k]) for k in glob_arrays)
    )
    ok = slot_named == 20 and slot_manifest == 20 and same
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'slot recorded':<24}"
        f" --capture-layer 20 records {slot_named}, --manifest --layer 20 records"
        f" {slot_manifest}, and the two trained the same parameters: {same}"
    )
    failures += not ok

    # A 2-D shard has no layer axis, so --layer can only name the slot. The output name lacks
    # `.npz`, which `np.savez` adds, and the printed path has to be the file it wrote.
    out = os.path.join(root, "glob_flat")
    proc = run_train(
        ["--activations", shards2, "--layer", 20, "--d-model", D_MODEL, *common, "--out", out]
    )
    wrote = out + ".npz"
    slot_flat = checkpoint(wrote)[0].get("capture_layer") if os.path.exists(wrote) else None
    _, slot_single, _ = trained("manifest_single", "--manifest", flat_manifest)
    ok = (
        proc.returncode == 0
        and slot_flat == 20
        and f"wrote {wrote}" in proc.stdout
        and slot_single == 20
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'2-D slot recorded':<24}"
        f" --layer 20 on a 2-D shard records {slot_flat} and prints the file it wrote;"
        f" a one-slot manifest with no --layer records {slot_single}"
    )
    failures += not ok

    refused = run_train(
        ["--activations", shards2, "--layer", 20, "--capture-layer", 21, "--d-model", D_MODEL,
         *common, "--out", os.path.join(root, "clash.npz")]
    )
    detected = refused.returncode != 0 and not os.path.exists(os.path.join(root, "clash.npz"))
    print(
        f"      control (--layer 20 and --capture-layer 21 on a 2-D shard):"
        f" exit {refused.returncode} -> {'refused' if detected else 'NOT REFUSED'}"
    )
    failures += not detected
    return failures


def abstract_params(cfg):
    f32 = jnp.float32
    return sae_lib.SAEParams(
        w_enc=jax.ShapeDtypeStruct((cfg.d_model, cfg.d_sae), f32),
        b_enc=jax.ShapeDtypeStruct((cfg.d_sae,), f32),
        w_dec=jax.ShapeDtypeStruct((cfg.d_sae, cfg.d_model), f32),
        b_dec=jax.ShapeDtypeStruct((cfg.d_model,), f32),
    )


def check_step_flops():
    """6. `step_flops` against XLA's count for the compiled training step.

    A dictionary 64 times wider than the stream, so the matmuls carry nearly every FLOP and the
    count has little else to hide in.
    """
    batch, d_model, expansion, k = 256, 128, 64, 8
    cfg = SAEConfig(d_model=d_model, expansion_factor=expansion, k=k)
    params = abstract_params(cfg)
    optimizer = train_lib.build_optimizer(TrainConfig())
    opt_state = jax.eval_shape(optimizer.init, params)
    x = jax.ShapeDtypeStruct((batch, d_model), jnp.float32)
    cost = train_lib.make_step(cfg, optimizer).lower(params, opt_state, x).compile().cost_analysis()
    if isinstance(cost, (list, tuple)):
        cost = cost[0]
    xla = float(cost["flops"])
    claimed = train_lib.step_flops(batch, d_model, cfg.d_sae, k)
    dense = 12 * batch * d_model * cfg.d_sae
    ok = abs(claimed / xla - 1.0) < 0.05
    detected = abs(dense / xla - 1.0) > 0.05
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'step flops':<24} step_flops={claimed:.4e}"
        f" XLA={xla:.4e} ratio={claimed / xla:.3f} at B={batch} d={d_model} m={cfg.d_sae} k={k}"
    )
    print(
        f"      control (the decoder counted as a dense matmul): ratio={dense / xla:.3f}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    return int(not ok) + int(not detected)


def check_selection_limit(root):
    """7. A batch past the int32 reach of one selection fails early and says what to use."""
    failures = 0
    d_model, expansion = 4_096, 256  # d_sae = 2**20
    optimizer = train_lib.build_optimizer(TrainConfig())
    for recall in (1.0, 0.95):
        cfg = SAEConfig(d_model=d_model, expansion_factor=expansion, k=100, recall_target=recall)
        params = abstract_params(cfg)
        opt_state = jax.eval_shape(optimizer.init, params)
        step = train_lib.make_step(cfg, optimizer)
        fits = sae_lib.selection_limit(recall) // cfg.d_sae

        def trace(batch):
            x = jax.ShapeDtypeStruct((batch, d_model), jnp.float32)
            try:
                jax.eval_shape(step, params, opt_state, x)
            except Exception as exc:  # noqa: BLE001 - the check reads what was raised
                return exc
            return None

        over = trace(fits + 1)
        ok = isinstance(over, ValueError) and f"Use a batch of {fits:,} or fewer" in str(over)
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {'selection limit':<24} recall={recall}"
            f" batch {fits + 1:,} at d_sae={cfg.d_sae:,} raises"
            f" {type(over).__name__ if over else None}: {over}"
        )
        failures += not ok
        at = trace(fits)
        print(
            f"      control (batch {fits:,}, the widest that fits): "
            f"{'traces' if at is None else f'RAISED {type(at).__name__}: {at}'}"
        )
        failures += at is not None

    # `train.py` says so before it reads a shard or compiles anything.
    shards = os.path.join(root, "caps2", "shard-*.npy")
    if not glob.glob(shards):
        write_capture(os.path.join(root, "caps2"), make_activations(seed=60, tokens=512)[0], (20,))
    base = ["--activations", shards, "--d-model", D_MODEL, "--k", 64, "--steps", 2]
    wide = run_train([*base, "--expansion-factor", 16_385, "--out", os.path.join(root, "wide.npz")])
    ok = wide.returncode != 0 and "Use a batch of 4,095 or fewer" in wide.stderr
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'train.py refuses early':<24} batch 4,096 at"
        f" d_sae={D_MODEL * 16_385:,}: exit {wide.returncode}"
    )
    failures += not ok
    fits = run_train([*base, "--expansion-factor", 16_384, "--out", os.path.join(root, "fits.npz")])
    detected = "Use a batch of" not in fits.stderr
    print(
        f"      control (d_sae={D_MODEL * 16_384:,}, which fits at batch 4,096): passes the"
        f" selection check -> {'detected' if detected else 'NOT DETECTED'}"
    )
    failures += not detected
    return failures


def sharded_abstract(cfg, optimizer, mesh, batch):
    """Shapes for the step's arguments with the shardings `train` gives them. Nothing allocated."""
    specs = {
        (cfg.d_model, cfg.d_sae): P(None, LATENT_AXIS),
        (cfg.d_sae,): P(LATENT_AXIS),
        (cfg.d_sae, cfg.d_model): P(LATENT_AXIS, None),
        (cfg.d_model,): P(),
        (): P(),
    }

    def attach(a):
        return jax.ShapeDtypeStruct(
            a.shape, a.dtype, sharding=NamedSharding(mesh, specs[tuple(a.shape)])
        )

    params = jax.tree.map(attach, abstract_params(cfg))
    opt_state = jax.tree.map(attach, jax.eval_shape(optimizer.init, params))
    x = jax.ShapeDtypeStruct((batch, cfg.d_model), jnp.float32, sharding=NamedSharding(mesh, P()))
    return params, opt_state, x


COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")


def compiled_collectives(text):
    """Every collective in compiled HLO text, as `(kind, dims)` per shape in its result type."""
    found = []
    for line in text.splitlines():
        _, eq, rhs = line.partition("=")
        if not eq:
            continue
        rhs = " " + rhs.strip()
        for kind in COLLECTIVES:
            for form in (f" {kind}(", f" {kind}-start("):
                if form not in rhs:
                    continue
                result = rhs.split(form)[0]
                for piece in result.split("]")[:-1]:
                    dims = piece.rpartition("[")[2]
                    found.append((kind, tuple(int(d) for d in dims.split(",") if d)))
    return found


def check_decode_collectives(mesh):
    """9. The compiled step on the mesh all-reduces partial sums, never the gathered rows."""
    d_model, expansion, k, batch = 256, 16, 16, 128
    cfg = SAEConfig(d_model=d_model, expansion_factor=expansion, k=k)
    optimizer = train_lib.build_optimizer(TrainConfig())
    args = sharded_abstract(cfg, optimizer, mesh, batch)

    def collectives(step):
        return compiled_collectives(step.lower(*args).compile().as_text())

    def listing(found):
        return ", ".join(f"{kind} {list(dims)} {4 * math.prod(dims):,} B" for kind, dims in found)

    with_mesh = collectives(train_lib.make_step(cfg, optimizer, mesh))
    without = collectives(train_lib.make_step(cfg, optimizer))
    reduced = [dims for kind, dims in with_mesh if kind == "all-reduce"]
    ok = (batch, d_model) in reduced and max(math.prod(d) for d in reduced) <= batch * d_model
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'decode all-reduce':<24} d_model={d_model}"
        f" d_sae={cfg.d_sae} k={k} batch={batch} on {sae_lib.latent_shards(mesh)} shards:"
        f" {listing(with_mesh)}"
    )
    detected = ("all-reduce", (k * batch, d_model)) in without
    print(
        f"      control (the step built without the mesh): {listing(without)}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    failures = int(not ok) + int(not detected)

    # The loss and every gradient through the `shard_map` decode against one device, the same
    # parameters and batch. The twin reads a different batch, so the tolerance separates something.
    small = SAEConfig(d_model=D_MODEL, expansion_factor=8, k=4)
    init = sae_lib.init_params(jax.random.key(3), small)
    x, other = (jnp.asarray(make_activations(seed=s, tokens=128)[0]) for s in (90, 91))

    def loss_and_grads(params, batch_x, on):
        return jax.value_and_grad(
            lambda p: sae_lib.reconstruction_loss(p, small, batch_x, mesh=on)[0]
        )(params)

    run = jax.jit(loss_and_grads, static_argnums=2)
    loss8, grads8 = run(sae_lib.shard_params(init, mesh), x, mesh)
    one = jax.device_put(init, jax.devices()[0])
    loss1, grads1 = run(one, x, None)
    _, grads_other = run(one, other, None)

    def rel(a, b):
        return max(
            float(np.max(np.abs(np.asarray(p) - np.asarray(q)))
                  / max(float(np.max(np.abs(np.asarray(q)))), 1e-12))
            for p, q in zip(a, b)
        )

    err = max(rel(grads8, grads1), abs(float(loss8) - float(loss1)) / abs(float(loss1)))
    ok = err < 1e-5
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'shard_map gradients':<24} loss {float(loss8):.6f} on"
        f" the mesh against {float(loss1):.6f} on one device, worst relative error {err:.2e}"
    )
    control_err = rel(grads8, grads_other)
    detected = control_err > 1e-3
    print(
        f"      control (one-device gradients on another batch): {control_err:.2e}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    return failures + int(not ok) + int(not detected)


def dot_precisions(text):
    """The `precision` of every `stablehlo.dot_general` in lowered StableHLO text."""
    found = []
    for line in text.splitlines():
        if "stablehlo.dot_general" not in line:
            continue
        if "precision = [" not in line:
            found.append("unset")
            continue
        found.append(line.split("precision = [", 1)[1].split("]", 1)[0])
    return found


def check_precision(params, cfg, x):
    """10. Every matmul in the SAE asks for HIGHEST, the precision the steering probe reads at."""
    failures = 0
    optimizer = train_lib.build_optimizer(TrainConfig())
    theta = jnp.zeros((cfg.d_sae,), jnp.float32)
    programs = {
        "training step": train_lib.make_step(cfg, optimizer).lower(
            params, optimizer.init(params), x
        ),
        "threshold fit": sae_lib.calibration_stats.lower(params, x, cfg),
        "JumpReLU forward": jax.jit(sae_lib.forward_jumprelu).lower(params, x, theta),
        "bias fold": jax.jit(sae_lib.fold_pre_encoder_bias).lower(params),
    }
    for name, lowered in programs.items():
        found = dot_precisions(lowered.as_text())
        ok = bool(found) and all(p == "HIGHEST, HIGHEST" for p in found)
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {'precision, ' + name:<24} {len(found)} dot_general(s)"
            f" at {sorted(set(found))}"
        )
        failures += not ok
    plain = jax.jit(lambda p, b: b @ p.w_enc + p.b_enc).lower(params, x)
    found = dot_precisions(plain.as_text())
    detected = any(p != "HIGHEST, HIGHEST" for p in found)
    print(
        f"      control (the encoder matmul with no precision): {sorted(set(found))}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    return failures + (not detected)


def check_nonfinite():
    """11. `train` stops on a non-finite activation, wherever in the stream it sits.

    The stream is the 28 batches `batches_needed` asks for at 24 steps and 4 calibration
    batches: the scale fit reads 0 to 15, training 0 to 23, calibration 24 to 27.
    """
    failures = 0
    cfg = SAEConfig(d_model=D_MODEL, expansion_factor=EXPANSION, k=K_TRUE)
    small = TrainConfig(
        steps=24, batch_size=64, learning_rate=3e-3, warmup_steps=4, calibration_batches=4,
        log_every=1_000,
    )
    total = train_lib.batches_needed(small)
    rows = make_activations(seed=70, tokens=total * small.batch_size)[0]
    mesh = Mesh(np.array(jax.devices()[:1]), (LATENT_AXIS,))

    def stream(index=None, value=np.nan):
        size = small.batch_size
        batches = [rows[i * size : (i + 1) * size].copy() for i in range(total)]
        if index is not None:
            batches[index][3, 5] = value
        return iter(batches)

    cases = [
        ("a NaN in the scale sample, batch 0", 0, np.nan),
        ("an infinity in the scale sample, batch 5", 5, np.inf),
        ("a NaN in a training batch, batch 18", 18, np.nan),
        ("a NaN in a calibration batch, batch 25", 25, np.nan),
    ]
    for name, index, value in cases:
        try:
            train(cfg, small, stream(index, value), mesh=mesh, log=lambda line: None)
        except ValueError as exc:
            said = str(exc)
        else:
            said = None
        ok = said is not None and "non-finite" in said
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {'non-finite input':<24} {name}: {said or 'TRAINED'}")
        failures += not ok

    result = train(cfg, small, stream(), mesh=mesh, log=lambda line: None)
    clean = all(bool(np.isfinite(np.asarray(p)).all()) for p in result.params) and bool(
        np.isfinite(np.asarray(result.threshold)).any()
    )
    print(
        f"      control (the same stream with no NaN or infinity): finite parameters and"
        f" {int(np.isfinite(np.asarray(result.threshold)).sum())} live latents"
        f" -> {'trains' if clean else 'FAILED'}"
    )
    return failures + (not clean)


def check_batch_count(root):
    """12. `train.py` counts the batches the scale fit reads before it counts the steps.

    At `--steps 4` the scale fit reads 16 batches and the steps train on 4 of them, so the run
    reads 16 + 64 = 80. A count of steps plus calibration batches says 68, and a 68-batch capture
    then calibrates on 52.
    """
    failures = 0
    batch = 32
    short = write_capture(
        os.path.join(root, "caps68"), make_activations(seed=80, tokens=68 * batch)[0], (20,)
    )
    enough = write_capture(
        os.path.join(root, "caps80"), make_activations(seed=81, tokens=80 * batch)[0], (20,)
    )
    common = [
        "--expansion-factor", EXPANSION, "--k", K_TRUE, "--steps", 4, "--warmup-steps", 1,
        "--batch-size", batch, "--shuffle-bytes", 1 << 20,
    ]
    out = os.path.join(root, "count68.npz")
    refused = run_train(["--manifest", short, *common, "--out", out])
    said = "training reads 80" in refused.stderr
    ok = refused.returncode != 0 and said and not os.path.exists(out)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'scale batches counted':<24} --steps 4 on 68 batches:"
        f" exit {refused.returncode}"
    )
    failures += not ok
    ran = run_train(["--manifest", enough, *common, "--out", os.path.join(root, "count80.npz")])
    calibrated = ran.returncode == 0 and '"calibration_batches": 64' in ran.stdout
    print(
        f"      control (--steps 4 on 80 batches): exit {ran.returncode}, calibrates on 64"
        f" batches: {calibrated} -> {'trains' if calibrated else 'FAILED'}"
    )
    return failures + (not calibrated)


def main() -> int:
    devices = jax.devices()
    if len(devices) < 8:
        print(f"FAIL: need 8 simulated devices, got {len(devices)}")
        return 1
    mesh = Mesh(np.array(devices[:8]), (LATENT_AXIS,))
    print(f"mesh: {LATENT_AXIS}={mesh.shape[LATENT_AXIS]} on {devices[0].platform}")

    cfg = SAEConfig(d_model=D_MODEL, expansion_factor=EXPANSION, k=K_TRUE)
    train_cfg = TrainConfig(
        steps=4_000,
        batch_size=256,
        learning_rate=3e-3,
        warmup_steps=200,
        calibration_batches=32,
        log_every=1_000,
        seed=0,
    )
    print(f"d_model={cfg.d_model} d_sae={cfg.d_sae} k={cfg.k} atoms={ATOMS}\n")

    # One generative model, split into a training pool and held-out tokens.
    tokens, truth = make_activations(seed=1)
    pool, held_out = tokens[:-4_096], jnp.asarray(tokens[-4_096:])

    # Training runs on one device, and check 4 covers the sharded path on all eight. XLA's CPU
    # backend runs every device program as a thread in one pool, and a few hundred collective
    # executions in a row starve it. That's a property of the simulated mesh, not of the SAE.
    result = train(
        cfg,
        train_cfg,
        batch_stream(pool, train_cfg.batch_size, seed=3),
        mesh=Mesh(np.array(devices[:1]), (LATENT_AXIS,)),
        log=lambda line: print(f"    {line}"),
    )
    params, theta = result.params, result.threshold
    print()

    # 0. The documented identities.
    failures = check_identities(params, cfg, held_out, mesh)

    # 1. Training improves reconstruction.
    untrained = sae_lib.init_params(jax.random.key(train_cfg.seed), cfg)
    start = fvu(untrained, cfg, held_out)
    end = fvu(params, cfg, held_out)
    ok = end < 0.25 * start
    print(f"  [{'PASS' if ok else 'FAIL'}] reconstruction   fvu {start:.4f} -> {end:.4f}")
    failures += not ok
    if start <= 0.2:
        print("      FAIL: the untrained SAE already reconstructs, so this proves nothing.")
        failures += 1

    # 2. JumpReLU reproduces BatchTopK. Every control runs the same gate.
    def conversion(candidate):
        jac, dif, l0_bt, l0_jr = code_agreement(params, cfg, held_out, candidate)
        jr = fvu(params, cfg, held_out, theta=candidate)
        return jac > 0.95 and dif < 0.07 and abs(jr - end) < 0.02, jac, dif, l0_bt, l0_jr, jr

    ok, jaccard, diff, l0_bt, l0_jr, end_jr = conversion(theta)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] conversion       jaccard={jaccard:.4f}"
        f"  recon_diff={diff:.4f}  l0 {l0_bt:.2f} -> {l0_jr:.2f}  fvu {end:.4f} -> {end_jr:.4f}"
    )
    failures += not ok

    finite = theta[jnp.isfinite(theta)]
    print(
        f"      theta: {int(finite.size)} live, {float(jnp.min(finite)):.4f}"
        f" to {float(jnp.max(finite)):.4f}"
    )
    controls = {
        "permuted across latents": theta[
            jnp.asarray(np.random.default_rng(11).permutation(cfg.d_sae))
        ],
        "rescaled by 0.75": theta * 0.75,
        "dead latents opened to 0": jnp.where(jnp.isfinite(theta), theta, 0.0),
    }
    for name, candidate in controls.items():
        passed, c_jaccard, c_diff, _, c_l0, _ = conversion(candidate)
        print(
            f"      control ({name}): jaccard={c_jaccard:.4f}"
            f" recon_diff={c_diff:.4f} l0={c_l0:.2f}"
            f" -> {'NOT DETECTED' if passed else 'detected'}"
        )
        if passed:
            print(f"      FAIL: {name} also passes, so the fit isn't being tested.")
            failures += 1

    # 3. The learned dictionary recovers the atoms that generated the data.
    best = recovery(params.w_dec, truth)
    random_dict = np.random.default_rng(5).normal(size=params.w_dec.shape)
    control = recovery(random_dict, truth)
    ok = best.mean() > 0.9 and best.min() > 0.7
    detected = control.mean() < 0.7
    print(
        f"  [{'PASS' if ok else 'FAIL'}] dictionary       recovery mean={best.mean():.4f}"
        f" min={best.min():.4f}  {int((best > 0.9).sum())}/{len(best)} atoms above 0.9"
    )
    failures += not ok
    print(
        f"      control (random dictionary): mean={control.mean():.4f}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: a random dictionary recovers the atoms too, so nothing was learned.")
        failures += 1

    # 4. Sharding the dictionary doesn't change the answer. The trained parameters are folded, so
    # the loss runs without the pre-encoder subtraction. The sharded run decodes in a `shard_map`.
    loss_fn = jax.jit(sae_lib.reconstruction_loss, static_argnums=(1, 3, 4))
    replicated = jax.device_put(params, NamedSharding(mesh, P()))
    sharded = sae_lib.shard_params(params, mesh)
    loss_rep, _ = loss_fn(replicated, cfg, held_out, False, None)
    loss_shd, _ = loss_fn(sharded, cfg, held_out, False, mesh)
    rel = float(abs(loss_shd - loss_rep) / jnp.maximum(abs(loss_rep), 1e-8))
    spec = sharded.w_dec.sharding.spec
    ok = rel < 1e-5 and spec[0] == LATENT_AXIS
    print(
        f"  [{'PASS' if ok else 'FAIL'}] sharding         w_dec spec={tuple(spec)}"
        f"  loss {float(loss_rep):.6f} vs {float(loss_shd):.6f}  rel={rel:.3e}"
    )
    failures += not ok

    # The control has to run on the mesh, or it says nothing about the sharded program.
    shards = [np.asarray(s.data) for s in sharded.w_dec.addressable_shards]
    rolled = jax.device_put(
        jnp.concatenate([jnp.asarray(s) for s in shards[1:] + shards[:1]], axis=0),
        sharded.w_dec.sharding,
    )
    loss_rolled, _ = loss_fn(sharded._replace(w_dec=rolled), cfg, held_out, False, mesh)
    detected = abs(float(loss_rolled) - float(loss_rep)) > 1e-3 * abs(float(loss_rep))
    print(
        f"      control (shards rotated on the mesh): loss={float(loss_rolled):.6f}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the sharded loss ignores what the shards hold.")
        failures += 1

    with tempfile.TemporaryDirectory() as root:
        # 5. What the checkpoint records as its capture slot.
        failures += check_capture_slot(root)
        # 6. The FLOP count behind the logged MFU.
        failures += check_step_flops()
        # 7. The int32 reach of one BatchTopK selection.
        failures += check_selection_limit(root)
        # 8. A bfloat16 forward pass over float32 master weights.
        failures += check_bf16_training(pool)
        # 9. What the training step all-reduces on the mesh.
        failures += check_decode_collectives(mesh)
        # 10. The precision of every matmul.
        failures += check_precision(params, cfg, held_out)
        # 11. A NaN or an infinity anywhere in the stream stops the run.
        failures += check_nonfinite()
        # 12. The batches train.py counts before it trains.
        failures += check_batch_count(root)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
