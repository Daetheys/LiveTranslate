# LiveTranslate: notes for agents

Live speech translator **French ⇄ Hungarian** (plus "any language → FR / HU"), with speaker
diarization, shown as a phone-style chat. Everything runs locally; no cloud APIs.

## Status

- `desktop/`: working Python reference app, tested on Windows 11 + RTX 3080 (CUDA). Not yet tested
  on macOS: `setup.sh` / `run.sh` and the MPS code path are written but unverified.
- Default models switched (2026-11-10) to the phone configuration from
  [docs/LOW_MEMORY.md](docs/LOW_MEMORY.md): Opus-MT + CAM++. Verified so far: loads (2.0 GB RAM)
  and translates/attributes a real French sentence end to end. **Not yet verified**: a full
  multi-speaker conversation with these defaults (the dev PC ran out of memory during that test),
  and the NLLB fallback for other languages (hit MemoryError on the dev PC; now fails gracefully).
- `ios/`: does not exist yet. The goal is a native iPhone 15 Pro app with bundled models: read
  [docs/IOS_PORT.md](docs/IOS_PORT.md) before starting.
- Pipeline, parameters and UI protocol: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Layout

```
desktop/
  app.py           window (pywebview) + JS bridge `Api`; CLI flags
  pipeline.py      Segmenter (VAD) → Transcriber (Whisper) → RoutedTranslator (Opus-MT, NLLB fallback)
                   → Diarizer (CamppEmbedder / WavlmEmbedder), Engine, SessionLog
  textfix.py       text clean-up rules (pure Python), tested by tests/test_text.py
  ui/index.html    the whole UI (HTML/CSS/JS, phone-sized), visual reference for iOS
  ui_demo.py       opens the UI with sample data, no models: use it for UI work
  convert_nllb.py  one-time NLLB → fp16 safetensors conversion (lower RAM at load)
  setup.ps1 / run.ps1 / run.cmd   Windows      setup.sh / run.sh   macOS / Linux
docs/              architecture, iOS plan, low-memory measurements (LOW_MEMORY.md)
tools/lowmem/      CPU int8 benchmark scripts behind LOW_MEMORY.md
```

## Commands

```bash
cd desktop
python -m unittest discover tests      # fast, no torch / models needed
python ui_demo.py                      # UI only (needs pywebview)
./setup.sh && ./run.sh                 # macOS / Linux full app (downloads ~2 GB of models on first run)
```
Windows: `setup.ps1` once, then double-click `run.cmd`. Useful flags: `--input-device N`
(`--list-devices`), `--mode auto|hu|fr`, `--mt opus|nllb|ollama`, `--speaker-model campplus|microsoft/wavlm-base-plus-sv`,
`--no-diarization`, `--log-dir DIR`, `--debug` (web inspector).

## Models (downloaded from Hugging Face on first run)

| Role | Model | Size |
|---|---|---|
| Speech recognition | `openai/whisper-large-v3-turbo` | 1.6 GB |
| Translation FR ⇄ HU | `Helsinki-NLP/opus-mt-fr-hu` + `opus-mt-hu-fr` | 2 × 0.3 GB |
| Translation, other languages | `facebook/nllb-200-distilled-1.3B` (CC-BY-NC 4.0: non-commercial), loaded only on first use | 5.5 GB fp32, 2.7 GB after `convert_nllb.py` |
| Speaker embeddings | CAM++ `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` from `csukuangfj/speaker-embedding-models` (sherpa-onnx, CPU) | 28 MB |
| VAD | Silero VAD (bundled in the `silero-vad` package) | 2 MB |

The previous defaults (NLLB for everything, WavLM speakers) remain available through the flags above.

## Gotchas

- **RAM**: the dev PC has crashed several times from memory exhaustion. With the current defaults
  one app instance uses ~2 GB RAM / ~5.3 GB committed (CUDA libraries); the old NLLB+WavLM setup
  committed ~8.6 GB. Check free commit before any model-loading run, never run two at once, kill
  leftover `app.py` processes first, and prefer the torch-free tests.
- Loading the original NLLB `.bin` in fp16 needs ~8 GB of free commit and crashed with an access
  violation inside transformers' threaded loader. `convert_nllb.py` avoids it; `run.ps1` uses the
  converted copy automatically.
- transformers **5.x** API (`dtype=`, not `torch_dtype=`). `output_scores=True` prints a harmless
  "not valid" warning but still returns scores.
- Windows dev machine: C: is full, so the venv, HF cache, models and logs live on `D:\`
  (`LIVETRANSLATE_DATA` overrides). `.ps1` files are blocked by default: launch with `run.cmd`.
- Whisper hallucinates subtitle credits on silence ("Sous-titrage Société Radio-Canada"), loops
  sentences, and translates instead of transcribing when it picks the wrong language.
  `textfix.py` and the speaker-language fix in `Engine._process` handle these; add new real
  failures to `tests/test_text.py`.
- NLLB formats short outputs like subtitles ("- Köszönjük. - Köszönöm."): `clean_translation`
  (applied to Opus-MT output too).
- The speaker threshold depends on the embedding model: 0.70 for CAM++, 0.80 for WavLM
  (`SPEAKER_THRESHOLDS`). Re-calibrate when changing models (`tools/lowmem/` has the clips script).

## Conventions

- Keep the desktop app as the behavioural reference: when a threshold or rule changes, update
  `docs/ARCHITECTURE.md` and the tests.
- Large files (models, `*.onnx`, `*.mlpackage`, `*.safetensors`, session logs, wav) stay out of git.
