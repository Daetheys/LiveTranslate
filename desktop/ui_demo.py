"""Opens the UI with sample conversation data and no models: for working on ui/index.html.

    python ui_demo.py          # needs only pywebview
"""
import json
import os
import threading
import time

import webview

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")

SAMPLE = [
    dict(src="fr", tgt="hu", speaker=1, text="Bonjour ! Tu as bien dormi ?",
         translation="Jó reggelt! Jól aludtál?", latency=0.8, uncertain=False),
    dict(src="hu", tgt="fr", speaker=2, text="Igen, köszönöm. És te?",
         translation="Oui, merci. Et toi ?", latency=0.7, uncertain=False),
    dict(src="fr", tgt="hu", speaker=1, text="Très bien. On part vers huit heures pour la gare ?",
         translation="Nagyon jól. Nyolc körül indulunk a pályaudvarra?", latency=0.9, uncertain=False),
    dict(src="hu", tgt="fr", speaker=3, text="Szerintem jobb lenne fél nyolckor, mert sok a forgalom.",
         translation="Je pense qu'il vaudrait mieux à sept heures et demie, parce qu'il y a beaucoup de circulation.",
         latency=1.1, uncertain=False),
    dict(src="hu", tgt="fr", speaker=2, text="Rendben.", translation="D'accord.", latency=0.5, uncertain=True),
]


class FakeApi:
    def __init__(self):
        self._on = False

    def config(self):
        return {"mode": "auto", "langs": {}, "threshold": 0.7, "diarization": True, "ready": True}

    def toggle(self):
        self._on = not self._on
        return self._on

    def set_mode(self, mode): pass
    def merge(self, src, dst): pass
    def new_conversation(self): pass
    def set_threshold(self, value): pass


def play(window):
    window.events.loaded.wait()
    time.sleep(0.5)
    for m in SAMPLE:
        window.evaluate_js(f"window.onEngine('final', {json.dumps(m)})")
        time.sleep(0.3)
    window.evaluate_js("window.onEngine('partial', " + json.dumps(dict(
        src="fr", tgt="hu", text="Et on prend un taxi ou", translation="És taxit veszünk vagy")) + ")")


if __name__ == "__main__":
    w = webview.create_window("LiveTranslate (demo)", UI, js_api=FakeApi(), width=420, height=860,
                              background_color="#0d0f13")
    webview.start(lambda: threading.Thread(target=play, args=(w,), daemon=True).start(), debug=False)
