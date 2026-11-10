# iOS port plan (iPhone 15 Pro)

Status: **not started.** Nothing below has been built or measured on a device yet; library
choices are researched candidates to verify first.

## Requirements (from the owner)

- Runs fully offline on an **iPhone 15 Pro** (A17 Pro, 8 GB RAM), models bundled in the app,
  works out of the box.
- **Best quality** model tier: Whisper large-v3-turbo for speech recognition (Hungarian is weak
  on smaller Whisper models).
- Same features as the desktop app: FR ⇄ HU auto mode, "any → HU / FR" modes, live partial
  preview, speaker diarization with rename / merge, ⚠ unsure marker.
- UI: follow `desktop/ui/index.html` (run `desktop/ui_demo.py` to see it with sample data).
  It was designed at the iPhone 15 Pro viewport (393 × 852 pt).

**Read [LOW_MEMORY.md](LOW_MEMORY.md) first**: measured memory, speed and accuracy of smaller
model combinations. Key results: Hungarian needs Whisper turbo; Opus-MT fr↔hu (0.23 GB) translates
as well as NLLB-600M (0.7 GB); turbo on CPU is too slow, so it must run on the Neural Engine.

## Recommended stack (not yet confirmed by the owner)

Native **Swift / SwiftUI**, iOS 17+. Flutter was considered (one codebase with a Windows build)
but would run the models slower on the phone.

| Stage | Desktop (reference) | iOS candidate | To verify |
|---|---|---|---|
| Audio + VAD | sounddevice + Silero VAD | AVAudioEngine (16 kHz mono) + Silero VAD ONNX (onnxruntime) or WhisperKit's VAD | same 0.6 s end-of-utterance behaviour |
| Speech recognition | whisper-large-v3-turbo (transformers, fp16) | **WhisperKit** (Core ML, runs on the Neural Engine) with large-v3-turbo | latency per utterance, language-probability access for the "speaker's usual language" fix, no-speech prob, avg log-prob |
| Translation | Opus-MT fr→hu + hu→fr (transformers on GPU); NLLB-1.3B for other languages | **Opus-MT int8** (CTranslate2 or ONNX Runtime; 2 × 76 MB, measured 0.23 GB RAM). NLLB-600M int8 only if the "any language" modes are kept | SentencePiece tokenizer in Swift |
| Speaker embeddings | CAM++ ONNX via sherpa-onnx | **same ONNX file** with sherpa-onnx's iOS build | threshold 0.70 should carry over (same model) |
| Text rules | `desktop/textfix.py` | port 1:1 to Swift | run the same cases as `desktop/tests/test_text.py` |

Memory budget to check early: Whisper turbo (compressed Core ML, ~0.6–1 GB) + Opus-MT int8 (0.23 GB) +
CAM++ (< 0.1 GB) + runtime overhead must fit iOS's per-app limit; the `com.apple.developer.kernel.increased-memory-limit`
entitlement may be needed, and may require a paid developer account.

Model files are ~2 GB: keep them **out of git**. Add a script that downloads / converts them into the
Xcode project's resources folder (gitignored), and document it.

## Building and installing

- Building needs a Mac with Xcode. The owner has no Mac yet and will set one up later.
- Free Apple ID: the app can be installed on the owner's own iPhone but expires after 7 days.
  Paid Apple Developer account ($99/year): no expiry, TestFlight.
- Suggested layout: `ios/LiveTranslate.xcodeproj`, `ios/LiveTranslate/…`, `ios/scripts/fetch_models.sh`.

## Suggested order of work

1. Verify the desktop app on the Mac (`desktop/setup.sh`, `desktop/run.sh`, MPS backend).
2. Spike: WhisperKit turbo on the iPhone, measure latency and memory on recorded FR and HU audio
   (desktop session logs give real test clips: `utterances.jsonl` + wav files).
3. Spike: Opus-MT int8 on the iPhone, compare with the desktop output on the same sentences.
4. Pipeline in Swift (VAD → ASR → MT → diarization) with the parameters in ARCHITECTURE.md.
5. SwiftUI screens matching `desktop/ui/index.html`.
