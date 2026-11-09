"""Text clean-up for Whisper and NLLB output. Pure Python (no torch), unit-tested in
tests/test_text.py with real failures; port these rules as-is to other platforms."""
import re
import zlib

# Whisper hallucinates subtitle credits from its training data on noise / near-silence.
HALLUCINATIONS = (
    "sous-titrage", "sous-titres", "amara.org", "merci d'avoir regardé",
    "köszönöm a figyelmet", "feliratok", "thanks for watching", "subtitles by",
)

COMPRESSION_MAX = 2.4  # gzip ratio above this = repetition loop (same as openai/whisper)

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
PHRASE_LOOP = re.compile(r"(\b.{4,}?)(?:[\s,.;]+\1\b){2,}", re.IGNORECASE)
DIALOGUE_DASH = re.compile(r"^\s*[-–—]\s*")
ALTERNATIVES = re.compile(r"\s+[-–—]\s+")


def split_sentences(text):
    return [s for s in SENTENCE_END.split(text) if s.strip()]


def is_hallucination(text):
    t = text.lower().strip(" .!?…")
    return len(t) < 2 or any(h in t for h in HALLUCINATIONS)


def compression_ratio(text):
    b = text.encode()
    return len(b) / len(zlib.compress(b)) if b else 0.0


def collapse_repeats(text):
    """'A. A. A. B' -> 'A. B' and 'x, x, x' -> 'x': Whisper's typical decoding loops."""
    text = PHRASE_LOOP.sub(r"\1", text)
    out = []
    for s in SENTENCE_END.split(text):
        if not out or s.strip().lower() != out[-1].strip().lower():
            out.append(s)
    return " ".join(out).strip()


def clean_translation(t):
    """NLLB learned subtitle formatting: short inputs come back as '- A' or '- A - B'
    (two alternative renderings). Keep the first rendering, without the dash."""
    t = DIALOGUE_DASH.sub("", t)
    parts = ALTERNATIVES.split(t)
    # Alternatives are each a complete short sentence ("Köszönjük. - Köszönöm."), unlike "Igen - nem."
    if len(parts) > 1 and all(len(p.split()) <= 4 and p.rstrip()[-1:] in ".!?…" for p in parts):
        return parts[0]
    return t
