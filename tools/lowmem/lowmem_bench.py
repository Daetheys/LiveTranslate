"""CPU-only int8 benchmark of a model combination (one combination per process).
Usage: python -I lowmem_bench.py <whisper dir> <nllb dir> <clips dir> <json out>"""
import glob, json, os, re, sys, time, unicodedata
import psutil

whisper_dir, nllb_dir, clips, out = sys.argv[1:5]
proc = psutil.Process()
def mem():
    m = proc.memory_info()
    return {"rss": m.rss / 1e9, "private": m.private / 1e9}

res = {"whisper": os.path.basename(whisper_dir), "nllb": os.path.basename(nllb_dir), "base": mem()}

from faster_whisper import WhisperModel, decode_audio
import ctranslate2, sentencepiece as spm

t = time.perf_counter()
asr = WhisperModel(whisper_dir, device="cpu", compute_type="int8", cpu_threads=4)
res["whisper_load_s"] = time.perf_counter() - t
res["after_whisper"] = mem()

t = time.perf_counter()
mt = ctranslate2.Translator(nllb_dir, device="cpu", compute_type="int8", intra_threads=4)
sp = spm.SentencePieceProcessor(model_file=os.path.join(nllb_dir, "sentencepiece.bpe.model"))
res["nllb_load_s"] = time.perf_counter() - t
res["after_nllb"] = mem()

CODES = {"fr": "fra_Latn", "hu": "hun_Latn"}
def translate(text, src, tgt):
    toks = [CODES[src]] + sp.encode(text, out_type=str) + ["</s>"]
    r = mt.translate_batch([toks], target_prefix=[[CODES[tgt]]], beam_size=4, no_repeat_ngram_size=4,
                           max_decoding_length=2 * len(toks) + 10)
    return sp.decode(r[0].hypotheses[0][1:])

import wave, numpy as np
import struct
def read_wav(path):  # PyAV 19 breaks faster_whisper.decode_audio; FLEURS wavs are float32 (format 3)
    b = open(path, "rb").read()
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ": fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data": data = body
        pos += 8 + size + (size & 1)
    tag, ch, rate, _, _, bits = fmt
    assert rate == 16000, rate
    a = np.frombuffer(data, np.float32 if tag == 3 else np.int16).astype(np.float32)
    if tag != 3: a /= 32768
    return a.reshape(-1, ch).mean(axis=1)

def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", " ", s).split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)

res["clips"] = []
errs = {"fr": [0, 0], "hu": [0, 0]}
for wav in sorted(glob.glob(os.path.join(clips, "*.wav"))):
    true_lang = os.path.basename(wav)[:2]
    ref = open(wav[:-4] + ".txt", encoding="utf-8").read().strip()
    audio = read_wav(wav)
    t = time.perf_counter()
    segs, info = asr.transcribe(audio, beam_size=5, condition_on_previous_text=False, vad_filter=False)
    text = " ".join(s.text.strip() for s in segs)
    probs = dict(info.all_language_probs or [])
    lang = max(("fr", "hu"), key=lambda c: probs.get(c, 0))
    if lang != info.language:  # restrict to FR/HU like the app's auto mode
        segs, info = asr.transcribe(audio, language=lang, beam_size=5, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs)
    asr_s = time.perf_counter() - t
    t = time.perf_counter()
    tr = translate(text, lang, "fr" if lang == "hu" else "hu")
    mt_s = time.perf_counter() - t
    e, n = wer(ref, text)
    errs[true_lang][0] += e; errs[true_lang][1] += n
    res["clips"].append({"clip": os.path.basename(wav), "audio_s": len(audio) / 16000, "lang": lang,
                         "lang_ok": lang == true_lang, "asr_s": asr_s, "mt_s": mt_s,
                         "wer": e / n, "text": text, "translation": tr})
m = proc.memory_info()
res["peak"] = {"rss": m.peak_wset / 1e9, "private": m.peak_pagefile / 1e9}
res["wer"] = {k: v[0] / v[1] for k, v in errs.items()}
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{res['whisper']} + {res['nllb']}: peak RAM {res['peak']['rss']:.2f} GB, peak private {res['peak']['private']:.2f} GB, "
      f"WER fr {res['wer']['fr']:.0%} hu {res['wer']['hu']:.0%}, "
      f"asr {sum(c['asr_s'] for c in res['clips'])/len(res['clips']):.1f}s mt {sum(c['mt_s'] for c in res['clips'])/len(res['clips']):.1f}s per clip "
      f"(avg audio {sum(c['audio_s'] for c in res['clips'])/len(res['clips']):.1f}s)")
