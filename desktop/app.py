"""LiveTranslate desktop: live speech translation (French <-> Hungarian) with speaker diarization.

The UI is ui/index.html (phone-sized, also the visual reference for the iOS port), shown in a
native window by pywebview. All audio and model work happens in pipeline.py.

    page --(window.pywebview.api.*)--> Api --> Engine
    page <--(window.onEngine(kind, payload))-- Engine.emit
"""
import argparse
import json
import os
import threading

import webview

from pipeline import LANGS, MODES, Engine

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")


class Api:
    """Methods callable from the page as `await window.pywebview.api.<name>(...)`.
    Attributes starting with "_" are not exposed to JavaScript."""

    def __init__(self, engine):
        self._engine = engine

    def config(self):
        e = self._engine
        return {"mode": e.mode, "langs": {k: v[0] for k, v in LANGS.items()},
                "threshold": e.speaker_threshold, "diarization": not e.args.no_diarization,
                "ready": e.ready.is_set()}

    def set_mode(self, mode):
        if mode in MODES:
            self._engine.mode = mode

    def toggle(self):
        """Starts or stops listening; returns True while listening. Raises on mic errors."""
        e = self._engine
        if e.running.is_set():
            e.stop()
            return False
        e.start()
        return True

    def new_conversation(self):
        if self._engine.diarizer:
            self._engine.diarizer.reset()

    def merge(self, src, dst):
        if self._engine.diarizer:
            self._engine.diarizer.merge(int(src), int(dst))

    def set_threshold(self, value):
        if self._engine.diarizer:
            self._engine.diarizer.threshold = float(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODES), default="auto",
                    help="auto = French <-> Hungarian; hu / fr = any language into Hungarian / French")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3-turbo",
                    help="try openai/whisper-large-v3 if Hungarian recognition is weak")
    ap.add_argument("--mt", choices=["opus", "nllb", "ollama"], default="opus",
                    help="opus = Opus-MT for FR<->HU, NLLB loaded only when another language "
                         "appears; nllb = NLLB for everything; ollama = local LLM")
    ap.add_argument("--opus-fr-hu", default="Helsinki-NLP/opus-mt-fr-hu")
    ap.add_argument("--opus-hu-fr", default="Helsinki-NLP/opus-mt-hu-fr")
    ap.add_argument("--nllb-model", default="facebook/nllb-200-distilled-1.3B",
                    help="HF id or local folder (see convert_nllb.py for a lighter fp16 copy)")
    ap.add_argument("--ollama-model", default="gemma3:4b")
    ap.add_argument("--speaker-model", default="campplus",
                    help="campplus (28 MB ONNX, default) or microsoft/wavlm-base-plus-sv")
    ap.add_argument("--speaker-threshold", type=float, default=None,
                    help="cosine similarity needed to match a known voice (higher = more speakers); "
                         "default 0.70 for campplus, 0.80 for wavlm")
    ap.add_argument("--no-diarization", action="store_true")
    ap.add_argument("--log-dir", default=None,
                    help="save every utterance (wav + utterances.jsonl) in a session folder here")
    ap.add_argument("--input-device", default=None,
                    help="sounddevice input index or name substring (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--debug", action="store_true", help="open the web inspector")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.input_device is not None and args.input_device.isdigit():
        args.input_device = int(args.input_device)

    window = None

    def emit(kind, payload):
        window.evaluate_js(f"window.onEngine({json.dumps(kind)}, {json.dumps(payload)})")

    engine = Engine(args, emit)
    window = webview.create_window("LiveTranslate", UI, js_api=Api(engine), width=420, height=860,
                                   min_size=(340, 560), background_color="#0d0f13")

    def load_models():
        window.events.loaded.wait()  # the page must exist before it receives status events
        try:
            engine.load()
        except Exception as e:
            emit("error", f"Loading failed: {type(e).__name__}: {e}")

    webview.start(lambda: threading.Thread(target=load_models, daemon=True).start(), debug=args.debug)
    engine.stop()


if __name__ == "__main__":
    main()
