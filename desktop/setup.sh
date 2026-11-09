#!/usr/bin/env bash
# macOS (Apple Silicon, uses the MPS GPU) / Linux setup. Creates desktop/.venv.
# NOT YET TESTED on macOS: see CLAUDE.md "Status".
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"
