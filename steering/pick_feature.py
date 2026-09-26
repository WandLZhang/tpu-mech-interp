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

"""List the live latents of a trained SAE, ranked by how often they fire on captured activations.

    python3 steering/pick_feature.py --sae sae_l20.npz --manifest caps/manifest.json | tee pick.log
    export "$(tail -n 1 pick.log)"

It prints the ranking, one JSON line per latent, and ends on one line a shell can export,
`FEATURE=7987`, which names the top pick. The `export` above sets `FEATURE` from it, and the
ranking stays on screen and in `pick.log`.

A dead latent has an infinite threshold. Conditional steering never fires it, and static steering
adds its decoder row to every token it steers, a direction the SAE never writes at inference, so
`from_sae.py` refuses to build a bank from one. Most latents die on short runs, so pick from this
list.

With `--manifest` it reads up to `--tokens` activations from the first shard, encodes them, and
ranks each live latent by the share of tokens where `W_enc x + b_enc` reaches its threshold. It
drops latents that fire on almost every token, which carry little. Without a manifest it ranks by
threshold, lowest first. Runs on the host CPU and never touches the TPU.

`--layer` is the capture slot the SAE was trained on. `sae/train.py` records that slot in the
checkpoint, so `--layer` defaults to it, and a `--layer` that names another slot exits, because
ranking on another slot's activations picks a latent for the wrong stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))


def load(path: str):
    """`(w_enc, b_enc, threshold, config)` from what `train.py` wrote. `config` is `{}` if absent."""
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in ("w_enc", "b_enc", "threshold") if k not in data]
        if missing:
            raise ValueError(f"{path} lacks {missing}; is it a train.py output?")
        return (
            np.asarray(data["w_enc"], np.float32),
            np.asarray(data["b_enc"], np.float32),
            np.asarray(data["threshold"], np.float32),
            json.loads(str(data["config"])) if "config" in data else {},
        )


def activations(manifest_path: str, layer: int, tokens: int) -> np.ndarray:
    from capture_activations import layer_index, manifest_shards, read_manifest

    manifest = read_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"no manifest at {manifest_path}")
    shard = np.load(manifest_shards(manifest_path)[0], mmap_mode="r")
    axis = layer_index(manifest, layer)
    rows = shard[:tokens] if axis is None else shard[:tokens, axis, :]
    return np.asarray(rows, np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sae", required=True, help="the .npz train.py wrote")
    ap.add_argument("--manifest", help="capture manifest to rank by firing rate")
    ap.add_argument(
        "--layer",
        type=int,
        help="capture slot the SAE was trained on; defaults to the slot the checkpoint records",
    )
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--max-rate", type=float, default=0.5, help="drop latents firing more often")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args(argv)

    w_enc, b_enc, theta, config = load(args.sae)
    trained = config.get("capture_layer")
    live = np.flatnonzero(np.isfinite(theta))
    print(f"{live.size} live of {theta.size} latents")
    if live.size == 0:
        print("no live latents; train longer or on more tokens", file=sys.stderr)
        return 1

    if args.manifest:
        layer = args.layer if args.layer is not None else trained
        if layer is None:
            ap.error(f"{args.sae} records no capture slot, so --manifest needs --layer")
        if trained is not None and int(trained) != layer:
            ap.error(
                f"--layer {layer} isn't the capture slot {args.sae} was trained on, which is "
                f"{trained}"
            )
        x = activations(args.manifest, layer, args.tokens)
        z = x @ w_enc[:, live] + b_enc[live]
        rate = (z >= theta[live]).mean(axis=0)
        keep = (rate > 0) & (rate <= args.max_rate)
        order = np.argsort(-rate[keep])
        picks = [
            {"feature": int(live[keep][j]), "fire_rate": round(float(rate[keep][j]), 5),
             "threshold": float(theta[live[keep][j]])}
            for j in order[: args.top]
        ]
        print(
            f"ranked by firing rate over {x.shape[0]} tokens of slot {layer}, "
            f"rate at most {args.max_rate}"
        )
    else:
        order = np.argsort(theta[live])
        picks = [{"feature": int(live[j]), "threshold": float(theta[live[j]])} for j in order[: args.top]]
        print("ranked by threshold, lowest first")

    for p in picks:
        print("  " + json.dumps(p))
    if not picks:
        print("no live latent fired within the rate bounds", file=sys.stderr)
        return 1
    # The last line, in the form `export "$(tail -n 1 pick.log)"` reads.
    print(f"FEATURE={picks[0]['feature']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
