# LiveTranslate: notes for agents

Live speech translator **French ⇄ Hungarian** (plus "any language → FR / HU"), with speaker
diarization, shown as a phone-style chat. Everything runs locally; no cloud APIs.

## Status

- `desktop/`: working Python reference app, tested on Windows 11 + RTX 3080 (CUDA). Not yet tested
  on macOS: `setup.sh` / `run.sh` and the MPS code path are written but unverified.
- `ios/`: does not exist yet. The goal is a native iPhone 15 Pro app with bundled models: read
  [docs/IOS_PORT.md](docs/IOS_PORT.md) before starting.
- Pipeline, parameters and UI protocol: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Layout

```
desktop/
  app.py           window (pywebview) + JS bridge `Api`; CLI flags
  pipeline.py      Segmenter (VAD) → Transcriber (Whisper) → NllbTranslator → Diarizer, Engine, SessionLog
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
./setup.sh && ./run.sh                 # macOS / Linux full app (downloads ~9 GB of models on first run)
```
Windows: `setup.ps1` once, then double-click `run.cmd`. Useful flags: `--input-device N`
(`--list-devices`), `--mode auto|hu|fr`, `--no-diarization`, `--log-dir DIR`, `--debug` (web inspector).

## Models (downloaded from Hugging Face on first run)

| Role | Model | Size |
|---|---|---|
| Speech recognition | `openai/whisper-large-v3-turbo` | 1.6 GB |
| Translation | `facebook/nllb-200-distilled-1.3B` (CC-BY-NC 4.0: non-commercial) | 5.5 GB fp32, 2.7 GB after `convert_nllb.py` |
| Speaker embeddings | `microsoft/wavlm-base-plus-sv` | 0.4 GB |
| VAD | Silero VAD (bundled in the `silero-vad` package) | 2 MB |

GPU memory in use ≈ 4.5 GB. Latency on the RTX 3080 ≈ 1 s per sentence after warm-up.

## Gotchas

- **RAM**: one app instance commits ~8.6 GB of system memory. The dev PC has crashed from memory
  exhaustion: never run two model-loading processes at once, kill leftover `app.py` processes
  before tests, and prefer the torch-free tests.
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
- NLLB formats short outputs like subtitles ("- Köszönjük. - Köszönöm."): `clean_translation`.

## Conventions

- Keep the desktop app as the behavioural reference: when a threshold or rule changes, update
  `docs/ARCHITECTURE.md` and the tests.
- Large files (models, `*.onnx`, `*.mlpackage`, `*.safetensors`, session logs, wav) stay out of git.
