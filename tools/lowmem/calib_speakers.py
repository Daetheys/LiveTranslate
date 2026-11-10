"""Speaker-threshold calibration for CAM++: real FLEURS clips split in halves (same speaker)
vs other clips. Usage: python -I calib_speakers.py <dir with clips/ and voices/> <speaker-embedding-models snapshot dir>"""
import glob, os, sys, struct, numpy as np, sherpa_onnx
sys.stdout.reconfigure(encoding="utf-8")
S, SNAP = sys.argv[1], sys.argv[2]
def read_wav(path):
    b = open(path, "rb").read(); pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos+4], struct.unpack("<I", b[pos+4:pos+8])[0]; body = b[pos+8:pos+8+size]
        if cid == b"fmt ": fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data": data = body
        pos += 8 + size + (size & 1)
    a = np.frombuffer(data, np.float32 if fmt[0] == 3 else np.int16).astype(np.float32)
    return (a if fmt[0] == 3 else a / 32768).reshape(-1, fmt[1]).mean(1)
# items: (speaker label, audio)
items = []
for f in sorted(glob.glob(f"{S}/clips/*.wav")):
    a = read_wav(f); h = len(a) // 2; spk = os.path.basename(f)[:-4]
    items += [(spk, a[:h]), (spk, a[h:]), (spk, a[h:h + 2 * 16000])]  # halves + a 2 s piece
tts = {"0": "Zira", "1": "Hazel", "2": "Zira", "3": "Hazel", "4": "Zira"}
for f in sorted(glob.glob(f"{S}/voices/*.wav")):
    items.append(("tts-" + tts[os.path.basename(f)[0]], read_wav(f)))
for name in ["3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"]:
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=os.path.join(SNAP, name), num_threads=2))
    E = []
    for _, a in items:
        s = ex.create_stream(); s.accept_waveform(sample_rate=16000, waveform=a); s.input_finished()
        e = np.array(ex.compute(s)); E.append(e / np.linalg.norm(e))
    E = np.stack(E); M = E @ E.T; lab = [l for l, _ in items]
    same = [M[i, j] for i in range(len(lab)) for j in range(i) if lab[i] == lab[j]]
    diff = [(M[i, j], lab[i], lab[j]) for i in range(len(lab)) for j in range(i) if lab[i] != lab[j]]
    d = sorted(diff, reverse=True)
    print(f"{name}: dim {E.shape[1]}")
    print(f"  same speaker: min {min(same):.2f} p10 {np.percentile(same,10):.2f} mean {np.mean(same):.2f}")
    print(f"  different   : max {d[0][0]:.2f} p99 {np.percentile([x[0] for x in diff],99):.2f} mean {np.mean([x[0] for x in diff]):.2f}; top pairs: " + ", ".join(f"{x[1]}/{x[2]} {x[0]:.2f}" for x in d[:4]))
    kind = ["half", "half", "2s"] * 10 + ["tts"] * 5
    lo = sorted((M[i, j], lab[i], kind[i], kind[j]) for i in range(len(lab)) for j in range(i) if lab[i] == lab[j])[:6]
    print("  lowest same-speaker pairs:", ", ".join(f"{l} {a}/{b} {m:.2f}" for m, l, a, b in lo))
    full = [M[i, j] for i in range(len(lab)) for j in range(i) if lab[i] == lab[j] and kind[i] != "2s" and kind[j] != "2s"]
    print(f"  same speaker, >=3 s segments only: min {min(full):.2f} mean {np.mean(full):.2f}")
    d2 = [x for x in diff if {x[1], x[2]} != {"hu3", "hu4"} and {x[1], x[2]} != {"fr2", "fr4"}]
    print(f"  different, excluding hu3/hu4 and fr2/fr4: max {max(x[0] for x in d2):.2f} p99 {np.percentile([x[0] for x in d2],99):.2f}")
