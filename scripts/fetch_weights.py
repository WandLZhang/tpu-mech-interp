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

"""Download a model's top-level weights and print the local directory to hand the engine.

    SNAP=$(python3 scripts/fetch_weights.py openai/gpt-oss-120b | awk '/^PATH /{print $2}')
    python3 scripts/capture_activations.py --model-path "$SNAP" ...

Pass the engine this local path, never the repo id. Some repos carry extra copies of the weights in
subdirectories (gpt-oss ships `original/` and `metal/`), and the engine's own fetch matches
`*.safetensors` at any depth, so from a repo id it pulls every copy. `HF_HUB_OFFLINE=1` doesn't
help: the offline check then wants those same files.

Set `HF_HOME=/dev/shm/hf` first. `scripts/bootstrap_tpu_vm.sh` writes it into `~/.tpu_env`,
because the boot disk flushes at about 11 MB/s and a download's writeback backlog stalls every
later read and write.

The `DOWNLOAD` line counts the bytes this run fetched. Its rate reads those bytes over the run's
seconds, so a model already in the cache prints `fetched=0` and no rate, and a rerun after a
killed fetch rates only what the rerun fetched. Before it fetches, the script deletes the
`.incomplete` files a killed fetch left in the repo's cache and prints a `PARTIALS` line.
`huggingface_hub` resumes a dropped connection inside one process, but a new process starts
those files from zero, so the partials only hold RAM. The `XET` line comes
first and says whether Xet is off, because `huggingface_hub` can label its progress bars
"Downloading bytes" and "Reconstructing" over Xet or plain HTTP.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time


def blobs_dir(repo: str) -> str:
    """This repo's blob directory in the Hugging Face cache, which may not exist yet."""
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    return os.path.join(
        constants.HF_HUB_CACHE, repo_folder_name(repo_id=repo, repo_type="model"), "blobs"
    )


def blob_sizes(repo: str) -> dict:
    """Bytes on disk per blob in this repo's Hugging Face cache, `.incomplete` partials included."""
    blobs = blobs_dir(repo)
    if not os.path.isdir(blobs):
        return {}
    return {name: os.path.getsize(os.path.join(blobs, name)) for name in os.listdir(blobs)}


def remove_partials(repo: str) -> tuple[int, int]:
    """Delete the `.incomplete` files a killed fetch left. Returns (files, bytes).

    `huggingface_hub` tags each attempt's partial files, so a new process never resumes them and
    fetches those files from zero. Left alone, they hold RAM in `/dev/shm`: a Nemotron 3 Super
    fetch killed on 2026-09-25 left six of them, 12,132,024,320 bytes. Run one fetch of a repo at
    a time, because this also deletes a partial that another fetch is still writing.
    """
    blobs = blobs_dir(repo)
    files = total = 0
    if not os.path.isdir(blobs):
        return 0, 0
    for name in sorted(os.listdir(blobs)):
        if name.endswith(".incomplete"):
            path = os.path.join(blobs, name)
            total += os.path.getsize(path)
            os.remove(path)
            files += 1
    return files, total


def fetched_bytes(before: dict, after: dict) -> int:
    """Bytes the download added: new blobs, and what a partial blob gained on its way to whole."""
    total = 0
    for name, size in after.items():
        had = before.get(name, before.get(name + ".incomplete", 0))
        total += max(size - had, 0)
    return total


def xet_line() -> str:
    from huggingface_hub import constants

    if constants.HF_HUB_DISABLE_XET:
        return (
            "XET off (HF_HUB_DISABLE_XET=1): the files come over plain HTTP. huggingface_hub can "
            'still name its bars "Downloading bytes" and "Reconstructing", Xet\'s words.'
        )
    return (
        "XET on: set HF_HUB_DISABLE_XET=1 first, as ~/.tpu_env does. With Xet on, a large repo can "
        "pull every byte and never finish the file."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="Hugging Face repo id, such as google/gemma-4-26B-A4B-it")
    ap.add_argument("--revision", default=None, help="branch, tag or commit; default main")
    ap.add_argument("--workers", type=int, default=8, help="parallel file downloads")
    args = ap.parse_args(argv)

    from huggingface_hub import snapshot_download

    print(xet_line(), flush=True)
    files, freed = remove_partials(args.repo)
    if files:
        print(
            f"PARTIALS removed {files} .incomplete file(s), {freed} bytes, that a killed fetch "
            f"left. huggingface_hub doesn't resume another process's partial files, so it fetches "
            f"those files from zero.",
            flush=True,
        )
    before = blob_sizes(args.repo)
    started = time.time()
    path = snapshot_download(
        args.repo,
        revision=args.revision,
        max_workers=args.workers,
        # *.py keeps tokenizer code, such as Kimi K3's tokenization_kimi.py and the encoding_k3.py
        # it imports. The engine and build_corpus.py load it with trust_remote_code. The engine's
        # own download also keeps *.tiktoken.
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "*.jinja",
            "*.model",
            "*.txt",
            "*.py",
            "*.tiktoken",
        ],
        ignore_patterns=["original/*", "metal/*"],
    )
    secs = time.time() - started
    fetched = fetched_bytes(before, blob_sizes(args.repo))
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    total = sum(os.stat(s).st_size for s in shards)
    if not shards:
        print(f"no top-level safetensors in {args.repo}", file=sys.stderr)
        return 1
    if fetched:
        rate = f"rate_MBps={fetched / 1e6 / max(secs, 1e-9):.1f}"
    else:
        rate = "every file was in the cache already, so there's no rate"
    print(
        f"DOWNLOAD {args.repo} secs={secs:.1f} bytes={total} fetched={fetched} files={len(shards)} "
        f"{rate}"
    )
    print(f"PATH {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
