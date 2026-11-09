"""Mic -> VAD segmentation -> Whisper ASR -> NLLB (or local LLM) translation, with speaker diarization."""
import datetime
import json
import os
import queue
import threading
import time
import urllib.request
import wave
from collections import Counter, deque

import numpy as np
import torch

from textfix import (COMPRESSION_MAX, clean_translation, collapse_repeats, compression_ratio,
                     is_hallucination, split_sentences)

SAMPLE_RATE = 16000
CHUNK = 512  # silero-vad expects 512-sample windows at 16 kHz (32 ms)
CHUNK_SEC = CHUNK / SAMPLE_RATE

# Whisper code -> (display name, NLLB/FLORES-200 code). Detection is restricted to these.
LANGS = {
    "fr": ("Français", "fra_Latn"),
    "hu": ("Magyar", "hun_Latn"),
    "en": ("English", "eng_Latn"),
    "de": ("Deutsch", "deu_Latn"),
    "es": ("Español", "spa_Latn"),
    "it": ("Italiano", "ita_Latn"),
    "pt": ("Português", "por_Latn"),
    "ro": ("Română", "ron_Latn"),
    "nl": ("Nederlands", "nld_Latn"),
    "pl": ("Polski", "pol_Latn"),
    "ru": ("Русский", "rus_Cyrl"),
    "uk": ("Українська", "ukr_Cyrl"),
}

# Mode -> (candidate source languages, fixed target or None for "the other one").
MODES = {
    "auto": (("fr", "hu"), None),
    "hu": (tuple(LANGS), "hu"),
    "fr": (tuple(LANGS), "fr"),
}

# Quality gates, same spirit as openai/whisper's transcribe() heuristics.
NO_SPEECH_DROP = 0.6      # no-speech prob above this...
LOGPROB_SILENCE = -1.0    # ...with avg token log-prob below this: noise, drop it
LOGPROB_UNCERTAIN = -0.8  # shown with a warning marker
LANG_SURE = 0.8           # below this, prefer the speaker's usual language


class Segmenter:
    """Cuts a 16 kHz stream into utterances with silero-vad; emits partial and final audio."""

    def __init__(self, on_partial, on_final, threshold=0.5, min_silence=0.6,
                 min_speech=0.3, max_speech=20.0, partial_every=1.0, pre_roll=0.3):
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad()
        self.on_partial, self.on_final = on_partial, on_final
        self.threshold = threshold
        self.silence_chunks = int(min_silence / CHUNK_SEC)
        self.min_speech_chunks = int(min_speech / CHUNK_SEC)
        self.max_chunks = int(max_speech / CHUNK_SEC)
        self.partial_chunks = int(partial_every / CHUNK_SEC)
        self.pre = deque(maxlen=int(pre_roll / CHUNK_SEC))
        self.buf, self.speaking, self.silence, self.since_partial = [], False, 0, 0

    @torch.inference_mode()
    def feed(self, chunk):
        p = self.vad(torch.from_numpy(chunk), SAMPLE_RATE).item()
        if not self.speaking:
            self.pre.append(chunk)
            if p >= self.threshold:
                self.speaking, self.buf, self.silence, self.since_partial = True, list(self.pre), 0, 0
                self.pre.clear()
            return

        self.buf.append(chunk)
        self.since_partial += 1
        self.silence = self.silence + 1 if p < self.threshold - 0.15 else 0

        ended = self.silence >= self.silence_chunks
        if ended or len(self.buf) >= self.max_chunks:
            if len(self.buf) - self.silence >= self.min_speech_chunks:
                self.on_final(np.concatenate(self.buf))
            # On a forced cut (too long) keep listening to the same speaker.
            self.buf, self.since_partial = [], 0
            if ended:
                self.speaking = False
                self.vad.reset_states()
        elif self.since_partial >= self.partial_chunks:
            self.since_partial = 0
            self.on_partial(np.concatenate(self.buf))


class Transcriber:
    def __init__(self, model_id, device, dtype):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device, self.dtype = device, dtype
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        tok = self.processor.tokenizer
        vocab = tok.get_vocab()
        self.sot = vocab["<|startoftranscript|>"]
        self.no_speech = vocab.get("<|nospeech|>", vocab.get("<|nocaptions|>"))
        self.lang_ids = {code: vocab[f"<|{code}|>"] for code in LANGS}

    @torch.inference_mode()
    def detect(self, audio, candidates):
        """Returns (features, no-speech prob, {lang: prob} over the candidates)."""
        feats = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        # One decoder step after <|startoftranscript|> gives both the language
        # distribution and the no-speech probability (same as openai/whisper).
        sot = torch.tensor([[self.sot]], device=self.device)
        logits = self.model(input_features=feats, decoder_input_ids=sot).logits[0, -1].float()
        no_speech = float(logits.softmax(-1)[self.no_speech]) if self.no_speech is not None else 0.0
        lang_probs = logits[[self.lang_ids[c] for c in candidates]].softmax(-1)
        return feats, no_speech, dict(zip(candidates, lang_probs.tolist()))

    @torch.inference_mode()
    def transcribe(self, feats, lang, final=True):
        """Returns (raw text, average token log-prob)."""
        beams = 5 if final else 1
        out = self.model.generate(feats, language=lang, task="transcribe", num_beams=beams,
                                  max_new_tokens=220, return_dict_in_generate=True,
                                  output_scores=True)
        scores = self.model.compute_transition_scores(
            out.sequences, out.scores, out.get("beam_indices"), normalize_logits=beams == 1)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        return text, float(scores[0].mean()) if scores.numel() else -10.0


class NllbTranslator:
    def __init__(self, model_id, device, dtype):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def __call__(self, text, src, tgt, final=True):
        # NLLB is trained on single sentences: long multi-sentence inputs get parts
        # dropped or looped, so translate sentence by sentence (one batch).
        sentences = split_sentences(text)
        self.tok.src_lang = LANGS[src][1]
        enc = self.tok(sentences, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(LANGS[tgt][1]),
            num_beams=4 if final else 1, no_repeat_ngram_size=4,
            max_new_tokens=2 * enc["input_ids"].shape[1] + 10)
        return " ".join(clean_translation(t) for t in self.tok.batch_decode(out, skip_special_tokens=True))


class OllamaTranslator:
    """Translation through a local LLM served by Ollama (e.g. gemma3:4b / gemma3:12b)."""

    def __init__(self, model, host="http://localhost:11434"):
        self.model, self.url = model, f"{host}/api/generate"

    def __call__(self, text, src, tgt, final=True):
        prompt = (f"Translate this {LANGS[src][0]} speech transcript into {LANGS[tgt][0]}. "
                  f"Reply with the translation only, no comments.\n\n{text}")
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["response"].strip()


class Diarizer:
    """Online speaker clustering: one voice embedding per utterance, matched by cosine
    similarity against each known speaker's running centroid.

    identify() only looks; commit() changes state, so utterances that Whisper later
    rejects as noise never create or skew a speaker."""

    def __init__(self, model_id, device, threshold=0.80, min_new_speaker=2.0):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.device, self.threshold, self.min_new = device, threshold, min_new_speaker
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.sums, self.langs, self.next_id = {}, {}, 1  # id -> embedding sum / Counter

    @torch.inference_mode()
    def _embed(self, audio):
        inp = self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.model(**inp).embeddings[0].float(), dim=-1)

    def identify(self, audio):
        """Returns (speaker id or None for a new voice, similarity, all similarities, embedding, reliable)."""
        e = self._embed(audio)
        # Short utterances give noisy embeddings: attach them to the closest
        # speaker, but never create a speaker or update a voice print from them.
        reliable = len(audio) / SAMPLE_RATE >= self.min_new
        with self.lock:
            sims = {}
            if self.sums:
                ids = list(self.sums)
                centroids = torch.nn.functional.normalize(torch.stack([self.sums[i] for i in ids]), dim=-1)
                sims = dict(zip(ids, (centroids @ e).tolist()))
                best = max(sims, key=sims.get)
                if sims[best] >= self.threshold or not reliable:
                    return best, sims[best], sims, e, reliable
            return None, None, sims, e, reliable

    def commit(self, sid, e, reliable, lang):
        with self.lock:
            if sid is None or sid not in self.sums:  # new voice (or merged away meanwhile)
                sid, self.next_id = self.next_id, self.next_id + 1
                self.sums[sid] = e.clone()
            elif reliable:
                self.sums[sid] += e
            self.langs.setdefault(sid, Counter())[lang] += 1
            return sid

    def usual_lang(self, sid):
        with self.lock:
            c = self.langs.get(sid)
            return c.most_common(1)[0][0] if c and sum(c.values()) >= 2 else None

    def merge(self, src, dst):
        with self.lock:
            if src in self.sums and dst in self.sums:
                self.sums[dst] += self.sums.pop(src)
                self.langs[dst] = self.langs.get(dst, Counter()) + self.langs.pop(src, Counter())


class SessionLog:
    """One folder per session: utterances.jsonl + one wav per final utterance."""

    def __init__(self, root):
        self.dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.n = 0

    def write(self, audio, record):
        self.n += 1
        name = f"{self.n:04d}.wav"
        with wave.open(os.path.join(self.dir, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        record = {"id": self.n, "wav": name, **record}
        with open(os.path.join(self.dir, "utterances.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Engine:
    """Owns the models and threads. `emit(kind, payload)` is called from worker threads."""

    def __init__(self, args, emit):
        self.args, self.emit = args, emit
        self.mode = args.mode
        self.audio_q = queue.Queue()
        self.job_q = queue.Queue()
        self.running = threading.Event()
        self.ready = threading.Event()
        self.stream = None
        self.diarizer = None
        self.log = None

    def load(self):
        a = self.args
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():  # Apple Silicon Macs
            device = "mps"
        else:
            device = "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.emit("status", f"Loading Whisper ({a.asr_model}) on {device}…")
        self.asr = Transcriber(a.asr_model, device, dtype)
        if a.mt == "ollama":
            self.mt = OllamaTranslator(a.ollama_model)
        else:
            self.emit("status", f"Loading NLLB ({a.nllb_model})…")
            self.mt = NllbTranslator(a.nllb_model, device, dtype)
        if not a.no_diarization:
            self.emit("status", f"Loading speaker model ({a.speaker_model})…")
            self.diarizer = Diarizer(a.speaker_model, device, a.speaker_threshold)
        if a.log_dir:
            self.log = SessionLog(a.log_dir)
        # The first GPU pass compiles kernels (~4 s): do it now, not on the first sentence.
        self.emit("status", "Warming up…")
        feats, _, _ = self.asr.detect(np.zeros(SAMPLE_RATE, np.float32), ("fr", "hu"))
        self.asr.transcribe(feats, "fr")
        self.mt("Bonjour.", "fr", "hu")
        if self.diarizer:
            self.diarizer._embed(np.zeros(2 * SAMPLE_RATE, np.float32))
        self.segmenter = Segmenter(lambda x: self.job_q.put(("partial", x)),
                                   lambda x: self.job_q.put(("final", x)))
        threading.Thread(target=self._vad_loop, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        self.ready.set()
        self.emit("ready", device)

    def start(self):
        import sounddevice as sd

        def callback(indata, frames, t, status):
            self.audio_q.put(indata[:, 0].copy())

        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     blocksize=CHUNK, device=self.args.input_device,
                                     callback=callback)
        self.stream.start()
        self.running.set()

    def stop(self):
        self.running.clear()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _vad_loop(self):
        while True:
            chunk = self.audio_q.get()
            if self.running.is_set() and len(chunk) == CHUNK:
                self.segmenter.feed(chunk)

    def _worker(self):
        while True:
            kind, audio = self.job_q.get()
            # Partials are only previews: skip stale ones if more work is queued.
            if kind == "partial" and not self.job_q.empty():
                continue
            try:
                self._process(kind, audio)
            except Exception as e:  # keep the app alive on a bad segment / Ollama down
                self.emit("error", f"{type(e).__name__}: {e}")

    def _process(self, kind, audio):
        t0 = time.perf_counter()
        final = kind == "final"
        candidates, target = MODES[self.mode]
        feats, no_speech, lang_probs = self.asr.detect(audio, candidates)
        lang = max(lang_probs, key=lang_probs.get)

        speaker = similarity = None
        sims, lang_from_speaker = {}, False
        if final and self.diarizer:
            speaker, similarity, sims, emb, reliable = self.diarizer.identify(audio)
            # Short or accented phrases confuse language ID; a known speaker's usual
            # language is a better guess when Whisper hesitates.
            usual = self.diarizer.usual_lang(speaker) if speaker else None
            if lang_probs[lang] < LANG_SURE and usual in lang_probs and usual != lang:
                lang, lang_from_speaker = usual, True

        raw, avg_logprob = self.asr.transcribe(feats, lang, final)
        text = collapse_repeats(raw)
        compression = compression_ratio(text)

        dropped = None
        if no_speech > NO_SPEECH_DROP and avg_logprob < LOGPROB_SILENCE:
            dropped = "no speech"
        elif is_hallucination(text):
            dropped = "hallucination"
        elif compression > COMPRESSION_MAX:
            dropped = "repetition"

        record = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                  "duration": round(len(audio) / SAMPLE_RATE, 2), "mode": self.mode,
                  "lang": lang, "lang_probs": {k: round(v, 3) for k, v in lang_probs.items()},
                  "lang_from_speaker": lang_from_speaker, "no_speech": round(no_speech, 3),
                  "avg_logprob": round(avg_logprob, 3), "compression": round(compression, 2),
                  "raw_text": raw, "text": text, "dropped": dropped}

        if dropped:
            if final:
                if self.log:
                    self.log.write(audio, record)
                self.emit("final", None)
            return

        src = lang
        tgt = target or next(c for c in candidates if c != src)
        translation = text if src == tgt else self.mt(text, src, tgt, final=final)
        if final and self.diarizer:
            speaker = self.diarizer.commit(speaker, emb, reliable, lang)
        latency = time.perf_counter() - t0

        payload = {"src": src, "tgt": tgt, "text": text, "translation": translation,
                   "speaker": speaker, "similarity": similarity,
                   "uncertain": avg_logprob < LOGPROB_UNCERTAIN or lang_from_speaker,
                   "latency": latency, "duration": len(audio) / SAMPLE_RATE}
        if final and self.log:
            self.log.write(audio, {**record, "tgt": tgt, "translation": translation,
                                   "speaker": speaker,
                                   "similarity": None if similarity is None else round(similarity, 3),
                                   "speaker_sims": {k: round(v, 3) for k, v in sims.items()},
                                   "latency": round(latency, 2)})
        self.emit(kind, payload)
