# Architecture

This describes the desktop reference implementation in `desktop/`. The iOS app should reproduce
the same pipeline, parameters and UI behaviour (see [IOS_PORT.md](IOS_PORT.md)).

## Pipeline

```
mic 16 kHz mono, 512-sample chunks (32 ms)
  │
  ▼
Segmenter (Silero VAD) ── every 1 s while speaking ──► "partial" job (preview, greedy, no speaker)
  │  utterance ends after 0.6 s of silence (or 12 s max)
  ▼
"final" job ──► worker thread (one at a time; stale partials are skipped)
  1. Whisper: one decoder step → language probs over the mode's candidates + no-speech prob
  2. Diarizer.identify(audio) → closest known speaker (read-only)
  3. language fix: if Whisper's language prob < 0.8, use that speaker's usual language
  4. Whisper transcribe (beam 5 for finals, greedy for partials) + average token log-prob
  5. textfix: collapse repeats → drop if hallucination / compression ratio > 2.4 /
     (no-speech > 0.6 and log-prob < -1.0)
  6. translate sentence by sentence (beam 4, no_repeat_ngram 4) → clean_translation:
     Opus-MT for fr↔hu; NLLB (loaded on first use) for any other pair
  7. Diarizer.commit → speaker id (only kept utterances create/update speakers)
  8. emit "final" + append to session log (wav + utterances.jsonl)
```

Code map (`desktop/pipeline.py`): `Segmenter`, `Transcriber`, `NllbTranslator`,
`OllamaTranslator` (optional local-LLM alternative), `Diarizer`, `SessionLog`, `Engine`.
Text rules are in `desktop/textfix.py` (pure Python, unit tested).

## Parameters

| What | Value | Where |
|---|---|---|
| VAD speech threshold | 0.5 (silence when < 0.35) | `Segmenter` |
| End of utterance | 0.6 s silence | `Segmenter.min_silence` |
| Min / max utterance | 0.3 s / 12 s | `Segmenter` |
| Partial preview interval | 1 s | `Segmenter.partial_every` |
| Pre-roll kept before speech | 0.3 s | `Segmenter.pre_roll` |
| Whisper beams | 5 final, 1 partial | `Transcriber.transcribe` |
| Language "sure" | ≥ 0.8 | `LANG_SURE` |
| Drop as noise | no-speech > 0.6 and avg log-prob < −1.0 | `NO_SPEECH_DROP`, `LOGPROB_SILENCE` |
| Show "⚠ unsure" | avg log-prob < −0.8, or language taken from speaker | `LOGPROB_UNCERTAIN` |
| Repetition | gzip compression ratio > 2.4 | `textfix.COMPRESSION_MAX` |
| Translation beams | 4 final, 1 partial; `no_repeat_ngram_size=4` | `MarianTranslator`, `NllbTranslator` |
| Speaker match | cosine ≥ 0.70 (CAM++) / 0.80 (WavLM) to a speaker centroid (UI slider 0.50–0.95) | `SPEAKER_THRESHOLDS` |
| New speaker needs | ≥ 2 s of audio (shorter → nearest known speaker) | `Diarizer.min_new` |

## Modes

| Mode | Candidate source languages | Target |
|---|---|---|
| `auto` (FR ⇄ HU) | fr, hu | the other one |
| `hu` (→ Magyar) | 12 languages in `LANGS` | hu |
| `fr` (→ Français) | 12 languages in `LANGS` | fr |

## Diarization (online)

Each final utterance → speaker embedding, L2-normalised. Default: CAM++ (3D-Speaker,
`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`, 192-d, 28 MB, via sherpa-onnx on CPU).
Alternative: WavLM x-vector (`microsoft/wavlm-base-plus-sv`, 512-d, 0.4 GB).
Each speaker keeps the **sum** of their embeddings; the centroid is its normalised value.
Match = best cosine to a centroid ≥ threshold; otherwise a new speaker (only if the utterance
is ≥ 2 s). Merge = add sums. Each speaker also keeps a count of their languages, used by step 3.
Calibration, CAM++ on 10 real FLEURS speakers (each recording split in halves + a 2 s piece):
same speaker ≥ 0.69 (mean 0.82 for ≥ 3 s pieces), different speakers mean 0.19, 99th percentile 0.74
(two higher pairs, 0.83–0.88, are probably the same FLEURS reader). 0.70 favours splitting one person
in two (fixable with Merge) over merging two people (not fixable).
WavLM on three synthetic voices: same voice 0.94–0.98, different voices 0.57–0.86.

## UI ↔ engine protocol

The UI (`desktop/ui/index.html`) is a single page; Python is only a bridge.

Engine → UI: `window.onEngine(kind, payload)`

| kind | payload |
|---|---|
| `status` | string, e.g. "Loading Whisper…" |
| `ready` | device string ("cuda", "mps", "cpu") |
| `error` | string |
| `partial` | `{src, tgt, text, translation, speaker: null, similarity: null, uncertain, latency, duration}` |
| `final` | same with `speaker` (int) and `similarity` (float, null for a new speaker), or `null` when the utterance was dropped |

UI → engine (`window.pywebview.api.*`, all return promises): `config()`, `set_mode(mode)`,
`toggle()` → bool listening (throws on mic error), `new_conversation()`, `merge(src, dst)`,
`set_threshold(value)`.

UI behaviour: speaker names, colours and the message list live in the page. Bubbles whose
source language is French sit on the right ("my side"), others on the left. Space toggles the mic.

## Session logs

`--log-dir DIR` creates `DIR/<YYYYMMDD-HHMMSS>/` with `0001.wav`, `0002.wav`… and
`utterances.jsonl`: one record per final utterance, including dropped ones, with
`lang_probs`, `no_speech`, `avg_logprob`, `compression`, `raw_text`, `text`, `dropped`,
`translation`, `speaker`, `speaker_sims`, `latency`. Use these to tune thresholds from real speech.
