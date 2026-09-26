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

"""Fixed-length text prompts from wikitext-103, cut with the target model's own tokenizer.

    python3 scripts/build_corpus.py --model MODEL_OR_PATH --prompts 400 --tokens 440 \
        --out prompts.jsonl

Every prompt re-encodes to `--tokens` tokens with the model's tokenizer, special tokens aside, so
every batch has the same shape: 8 prompts of 440 tokens is 3,520, which the engine prefills over
four passes of its 1,024-token bucket. The engine adds a model's special tokens on top, so a Gemma
prompt reaches it at 441 with its BOS.

Paragraphs join with a blank line between them, and the text is cut at a token boundary. A cut
counts only if its text re-encodes to the same token ids. A cut that splits a character, or whose
edge tokens merge another way on their own, fails that test, and the cut then starts one token
later. The script re-encodes every prompt before it writes, and exits 1 if any comes back at
another length.

Short prompts starve capture. Before the layer filter, prompts of about 14 tokens ran at 167
tokens/s where 440-token prompts on the same slice ran at 2,534, because the fixed cost of each
engine call dominates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

# What goes between two paragraphs. Without it a paragraph's last word runs into the next one's
# first, as in ` " Calamaty Raven " .The game began`.
SEPARATOR = "\n\n"
# Wikitext rows shorter than this are headings, blank lines and stray fragments.
MIN_CHARS = 40
# What decode writes for the bytes of a character a cut split.
REPLACEMENT = "�"


def cut_prompts(
    paragraphs: Iterable[str], tokenizer, count: int, tokens: int, separator: str = SEPARATOR
) -> tuple:
    """Up to `count` prompts of `tokens` tokens each, cut from `paragraphs` in order.

    Returns `(prompts, skipped)`, where `skipped` counts the tokens passed over because a cut
    starting there didn't re-encode to its own ids.
    """
    prompts, skipped = [], 0
    text = ""
    for paragraph in paragraphs:
        if len(prompts) >= count:
            break
        text = f"{text}{separator}{paragraph}" if text else paragraph
        ids = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) >= tokens and len(prompts) < count:
            cut = ids[:tokens]
            piece = tokenizer.decode(cut)
            if tokenizer.encode(piece, add_special_tokens=False) == cut:
                prompts.append(piece)
                ids = ids[tokens:]
            else:
                skipped += 1
                ids = ids[1:]
        # What's left starts where the last cut ended, or inside a character a skip split. The
        # replacement character that decode puts there would reach the next prompt as text.
        text = tokenizer.decode(ids).lstrip(REPLACEMENT) if ids else ""
    return prompts, skipped


def paragraphs_from(rows: Iterable[dict], min_chars: int = MIN_CHARS) -> Iterable[str]:
    """The stripped text of each dataset row long enough to be a paragraph."""
    for row in rows:
        text = row["text"].strip()
        if len(text) >= min_chars:
            yield text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="repo id or local path; supplies the tokenizer")
    ap.add_argument("--prompts", type=int, default=400, help="how many prompts to write")
    ap.add_argument("--tokens", type=int, default=440, help="tokens per prompt")
    ap.add_argument("--out", required=True, help="destination .jsonl, one {\"text\": ...} a line")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Kimi K3 ships its tokenizer as Python code. The engine loads it with trust_remote_code too.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = load_dataset(args.dataset, args.config, split="train", streaming=True)
    out, skipped = cut_prompts(paragraphs_from(rows), tok, args.prompts, args.tokens)
    lengths = [len(tok.encode(text, add_special_tokens=False)) for text in out]
    wrong = [n for n, length in enumerate(lengths) if length != args.tokens]

    with open(args.out, "w", encoding="utf-8") as fh:
        for text in out:
            fh.write(json.dumps({"text": text}) + "\n")
    print(
        f"CORPUS prompts={len(out)} tokens_per_prompt={args.tokens} "
        f"reencoded_ok={len(out) - len(wrong)} skipped_tokens={skipped} -> {args.out}",
        flush=True,
    )
    for n in wrong:
        print(f"prompt {n} re-encodes to {lengths[n]} tokens, not {args.tokens}", flush=True)
    # The streaming reader leaves a thread that never joins, so a normal exit hangs.
    os._exit(0 if len(out) == args.prompts and not wrong else 1)


if __name__ == "__main__":
    sys.exit(main())
