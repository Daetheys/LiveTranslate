# Low-memory model options (measured 2026-11-10)

Question: can the pipeline run within a phone's memory? Each model combination ran **CPU only,
int8, 4 threads** (CTranslate2 / faster-whisper, no torch) on 10 real FLEURS test recordings
(5 French, 5 Hungarian, 6–23 s each, 12.4 s on average). Scripts: `tools/lowmem/`.

## Speech recognition + translation

| Whisper (int8) | Translation (int8) | Peak RAM | WER French | WER Hungarian | Whisper time per clip |
|---|---|---|---|---|---|
| base | NLLB-600M | 1.17 GB | 29 % | **66 %** | 2.1 s |
| small | NLLB-600M | 1.37 GB | 16 % | **43 %** | 4.3 s |
| large-v3-turbo | NLLB-600M | 1.98 GB | 9 % | 9 % | 15.5 s |
| large-v3-turbo | NLLB-1.3B | 3.51 GB | 9 % | 9 % | 15.6 s |
| large-v3-turbo | Opus-MT fr↔hu | ≈ 1.25 GB (estimate: turbo 0.99 + Opus 0.23, measured separately) | 9 % | 9 % | 15.5 s |

RAM = peak working set (physical memory). On Windows the *committed* memory was 3–4× higher
(CTranslate2 reserves memory it never touches); iOS counts the physical footprint, but verify on device.

Model RAM after loading: Whisper base 0.19 GB, small 0.38 GB, turbo 0.99 GB; NLLB-600M +0.7 GB,
NLLB-1.3B +1.5 GB, Opus-MT both directions 0.23 GB.

## Findings

1. **Hungarian needs Whisper turbo.** Base and small mangle Hungarian (43–66 % word error rate)
   and the translations of their output are nonsense. Turbo: 9 % on both languages.
2. **Opus-MT (Helsinki-NLP/opus-mt-fr-hu and opus-mt-hu-fr) is the best translation option for
   a phone**: 2 × 76 MB on disk, 0.23 GB RAM, 0.15 s per sentence, quality on par with
   NLLB-600M (better on 3 of 5 FR→HU sentences, mixed on HU→FR). It only covers FR↔HU, so the
   "any language →" modes would need NLLB or more Opus pairs. Licence: check the model cards
   (NLLB is non-commercial).
3. NLLB-1.3B costs +1.5 GB for a small quality gain over 600M: not worth it on a phone.
4. **CPU is too slow for turbo**: 15.5 s to transcribe 12.4 s of audio on 4 desktop cores.
   On iPhone the encoder must run on the Neural Engine (WhisperKit Core ML). WhisperKit publishes
   compressed turbo builds (`openai_whisper-large-v3-v20240930_turbo_632MB` in
   `argmaxinc/whisperkit-coreml`). Its speed and accuracy on Hungarian are **not measured yet**.

## Suggested phone configuration (now the desktop default too)

| Part | Model | Approx. RAM |
|---|---|---|
| Speech | Whisper large-v3-turbo, compressed Core ML (WhisperKit, ~632 MB) | ~1 GB (to verify) |
| Translation | Opus-MT fr→hu + hu→fr, int8 | 0.23 GB |
| Speakers | small speaker-embedding model (e.g. CAM++ ~30 MB via sherpa-onnx) instead of WavLM (0.4 GB) | < 0.1 GB; threshold 0.70 (calibrated, see ARCHITECTURE.md) |
| VAD | Silero | negligible |
| **Total** | | **≈ 1.3–1.5 GB** |

Also implemented in the desktop app: NLLB loads only when a language other than FR/HU appears;
greedy decoding for partial previews; utterances capped at 12 s. Still to do on the phone: prefer
runtimes that memory-map weights.
