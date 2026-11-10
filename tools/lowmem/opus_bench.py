import json, sys, time, psutil, ctranslate2, sentencepiece as spm
sys.stdout.reconfigure(encoding="utf-8")
bench, M = sys.argv[1], sys.argv[2]
p = psutil.Process()
tr = {}
for pair in ("fr-hu", "hu-fr"):
    d = f"{M}/opus-mt-{pair}"
    tr[pair] = (ctranslate2.Translator(d, device="cpu", compute_type="int8", intra_threads=4),
                spm.SentencePieceProcessor(model_file=d + "/source.spm"), spm.SentencePieceProcessor(model_file=d + "/target.spm"))
print(f"RSS after loading both directions: {p.memory_info().rss/1e9:.2f} GB")
r = json.load(open(bench, encoding="utf-8"))
ts = []
for c in r["clips"]:
    pair = "fr-hu" if c["lang"] == "fr" else "hu-fr"
    t, s_sp, t_sp = tr[pair]
    t0 = time.perf_counter()
    out = t.translate_batch([s_sp.encode(c["text"], out_type=str) + ["</s>"]], beam_size=4)
    ts.append(time.perf_counter() - t0)
    print(f"{c['clip']}\n  NLLB-600M: {c['translation']}\n  Opus-MT : {t_sp.decode(out[0].hypotheses[0])}")
m = p.memory_info()
print(f"peak RSS {m.peak_wset/1e9:.2f} GB, avg {sum(ts)/len(ts):.2f}s per sentence")
