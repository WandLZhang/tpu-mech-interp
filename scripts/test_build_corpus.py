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

"""Gate for `build_corpus.py`'s cut, on a real tokenizer. No network, no engine.

    python3 scripts/test_build_corpus.py

The tokenizer is the byte-level BPE `cpu_engine.write_tokenizer` trains, the one the engine gates'
checkpoint carries. The paragraphs mix its word list with characters past ASCII, which it splits
into two or three byte tokens, so some cut lands inside a character. `cut_prompts` has to write
prompts that each re-encode to the same token count, with a blank line at every paragraph seam and
no replacement character anywhere.

The control is the cut `build_corpus.py` made before: stripped paragraphs' ids run together with
nothing between them, and each cut written as decoded text. Each check runs on both, and a control
that passes fails the run.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_engine  # noqa: E402
from build_corpus import REPLACEMENT, SEPARATOR, cut_prompts  # noqa: E402

PROMPTS, TOKENS = 30, 40
failures = 0


def report(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    failures += not ok


def control(detected, text):
    global failures
    print(f"      control ({text}): {'detected' if detected else 'NOT DETECTED'}", flush=True)
    if not detected:
        print("      FAIL: the control didn't fail, so this check proves nothing.")
        failures += 1


def paragraphs(seed=3, count=80):
    """Sentences from the tokenizer's word list, a third of them holding characters past ASCII."""
    rng = random.Random(seed)
    extra = ["café", "naïve", "—", "日本", "Zürich"]
    out = []
    for _ in range(count):
        words = [rng.choice(cpu_engine.WORDS) for _ in range(rng.randint(12, 40))]
        if rng.random() < 0.34:
            for _ in range(rng.randint(1, 3)):
                words.insert(rng.randrange(len(words)), rng.choice(extra))
        out.append(" ".join(words).capitalize() + " .")
    return out


def old_cut(paragraphs, tokenizer, count, tokens):
    """The cut before the fix: ids run together with no separator, each cut decoded as it falls."""
    buf, out = [], []
    for text in paragraphs:
        buf.extend(tokenizer.encode(text.strip(), add_special_tokens=False))
        while len(buf) >= tokens and len(out) < count:
            chunk, buf = buf[:tokens], buf[tokens:]
            out.append(tokenizer.decode(chunk))
        if len(out) >= count:
            break
    return out


def glued(prompts, paragraphs):
    """Seams where a paragraph's end runs straight into the next one's start, over all prompts."""
    seams = [a[-8:] + b[:8] for a, b in zip(paragraphs, paragraphs[1:])]
    return sum(p.count(seam) for p in prompts for seam in seams)


def lengths(prompts, tokenizer):
    return [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]


root = tempfile.mkdtemp(prefix="build-corpus-test-")
try:
    from transformers import AutoTokenizer

    cpu_engine.write_tokenizer(os.path.join(root, "tok"))
    tok = AutoTokenizer.from_pretrained(os.path.join(root, "tok"))
    text = paragraphs()
    wide = sum(len(tok.encode(ch, add_special_tokens=False)) > 1 for ch in "éï—日ü")
    report(wide == 5, f"the tokenizer splits each of the 5 characters past ASCII into byte tokens")

    new, skipped = cut_prompts(text, tok, PROMPTS, TOKENS)
    old = old_cut(text, tok, PROMPTS, TOKENS)
    print(f"cut {len(new)} prompts of {TOKENS} tokens, {skipped} token(s) skipped; the old cut "
          f"made {len(old)}", flush=True)

    got = lengths(new, tok)
    report(len(new) == PROMPTS and set(got) == {TOKENS},
           f"all {len(new)} prompts re-encode to {TOKENS} tokens: {sorted(set(got))}")
    was = lengths(old, tok)
    control(set(was) != {TOKENS},
            f"the old cut, whose prompts re-encode to {sorted(set(was))} tokens")

    spanning = sum(SEPARATOR in p for p in new)
    report(glued(new, text) == 0 and spanning > 0,
           f"no paragraph runs into the next, and {spanning} prompts carry a blank line where "
           f"two paragraphs meet")
    control(glued(old, text) > 0, f"the old cut, with {glued(old, text)} seams glued shut")

    report(not any(REPLACEMENT in p for p in new),
           "no prompt holds a replacement character, so no cut split a character")
    control(any(REPLACEMENT in p for p in old),
            f"the old cut, with {sum(REPLACEMENT in p for p in old)} prompts holding one")

    few, _ = cut_prompts(text, tok, 3, TOKENS)
    report(few == new[:3], "asking for 3 prompts returns the first 3 of the same cut")
    none, _ = cut_prompts(text[:1], tok, PROMPTS, 10**6)
    report(none == [], "text shorter than one prompt gives no prompt rather than a short one")
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"FAILED: {failures} check(s)")
    raise SystemExit(1)
print("All checks passed.")
