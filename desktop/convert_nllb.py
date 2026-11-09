"""One-time conversion of an NLLB checkpoint to fp16 safetensors.

The original facebook/nllb-200-distilled-1.3B ships a 5.5 GB fp32 pytorch_model.bin; loading it
in fp16 needs ~8 GB of free commit (RAM + page file). The fp16 safetensors copy is 2.7 GB and is
memory-mapped at load time.

Usage (HF_HOME must point at the cache that holds the model):
    python convert_nllb.py facebook/nllb-200-distilled-1.3B D:\\models\\nllb-200-distilled-1.3B-fp16
"""
import gc
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file


def main(repo, out):
    src = snapshot_download(repo, allow_patterns=["*.json", "*.model", "pytorch_model.bin"])
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".json", ".model")):
            shutil.copy(os.path.join(src, f), out)
    # Plain load (not mmap): memory-mapped reads of this file crashed on this machine.
    sd = torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    tensors, seen = {}, set()
    for k in list(sd):
        v = sd.pop(k)
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape))
        if key not in seen:  # tied embeddings: store once, transformers re-ties them
            seen.add(key)
            tensors[k] = v.to(torch.float16).contiguous()
        del v
    gc.collect()
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    print(f"saved {len(tensors)} tensors to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
