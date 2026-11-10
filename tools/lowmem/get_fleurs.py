import csv, io, os, sys, tarfile, urllib.request
out = sys.argv[1]
base = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{}/"
for lang in ("hu_hu", "fr_fr"):
    tsv = urllib.request.urlopen(base.format(lang) + "test.tsv").read().decode()
    text = {r[1]: r[2] for r in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)}
    n = 0
    with urllib.request.urlopen(base.format(lang) + "audio/test.tar.gz") as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if not m.isfile() or name not in text: continue
                data = tar.extractfile(m).read()
                dst = os.path.join(out, f"{lang[:2]}{n}.wav")
                open(dst, "wb").write(data)
                open(dst[:-4] + ".txt", "w", encoding="utf-8").write(text[name])
                n += 1
                if n == 5: break
    print(lang, n, "clips")
