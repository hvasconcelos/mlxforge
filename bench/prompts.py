"""Deterministic, tokenizer-free prompt synthesis.

Prompts are seeded word streams sized in *words*; a per-server two-point
calibration (one short and one long non-streaming probe, reading
`usage.prompt_tokens`) solves tokens-per-word and the fixed chat-template
overhead for that engine's tokenizer, so target token counts land within a few
percent without any Python tokenizer dependency. The achieved
`usage.prompt_tokens` is recorded per request anyway, so residual error is
visible, not silent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# A fixed mid-frequency wordlist: common enough to tokenize predictably, varied
# enough that synthesized prompts don't collapse into repeated token runs.
_WORDS = (
    "river mountain copper lantern harvest meadow timber anchor compass marble "
    "garden whistle saddle barrel craft ribbon hammer voyage candle orchard "
    "bridge falcon cellar pebble drift canvas mirror saddle copper village "
    "window timber harbor meadow signal lantern basket runner copper anchor "
    "quarry beacon cobble thicket paddle vessel slate ember willow granite"
).split()

# The task framing is open-ended on purpose: engines should hit max_tokens, not
# EOS, so throughput numbers compare like with like (llama.cpp additionally
# gets ignore_eos from its adapter).
_TASK = (
    "Continue this never-ending inventory list, one numbered item per line, "
    "without ever stopping or concluding. Reference words: "
)


def synth_words(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


@dataclass(frozen=True)
class Calibration:
    tokens_per_word: float
    overhead_tokens: float  # chat template + task framing, in tokens

    def words_for(self, target_tokens: int) -> int:
        return max(8, round((target_tokens - self.overhead_tokens) / self.tokens_per_word))


def calibration_from_probes(w1: int, t1: int, w2: int, t2: int) -> Calibration:
    # Two probes of w1 < w2 words measured at t1/t2 prompt tokens: the slope is
    # tokens-per-word, the intercept is the fixed per-request overhead.
    tpw = (t2 - t1) / (w2 - w1)
    if tpw <= 0:  # degenerate usage reporting; fall back to ~1.3 tokens/word
        return Calibration(tokens_per_word=1.3, overhead_tokens=30.0)
    return Calibration(tokens_per_word=tpw, overhead_tokens=max(0.0, t1 - w1 * tpw))


CALIBRATION_PROBE_WORDS = (100, 400)


def probe_text(n_words: int, seed: int = 99) -> str:
    return _TASK + synth_words(n_words, seed)


def synth_prompt(target_tokens: int, cal: Calibration, seed: int) -> str:
    return _TASK + synth_words(cal.words_for(target_tokens), seed)


def unique_prefix(seed: int, cal: Calibration, tokens: int = 16) -> str:
    # Per-request unique lead-in so prompt-prefix caches can't serve scenario
    # traffic that is meant to measure prefill.
    return f"(session {seed}) " + synth_words(cal.words_for(tokens), seed) + " "
