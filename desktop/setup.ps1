# Windows + NVIDIA GPU setup.
# Everything heavy (venv ~6 GB, pip temp/cache) goes under $Data, default D:\ (C: was full on the dev
# machine). Override with:  .\setup.ps1 -Data E:\   or   $env:LIVETRANSLATE_DATA = "E:\"
param([string]$Data = $(if ($env:LIVETRANSLATE_DATA) { $env:LIVETRANSLATE_DATA } else { "D:\" }))
$ErrorActionPreference = "Stop"

$Venv = Join-Path $Data "venvs\livetranslate"
$env:TEMP = $env:TMP = Join-Path $Data "tmp"
$env:PIP_CACHE_DIR = Join-Path $Data "tmp\pip-cache"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

python -m venv $Venv
$py = Join-Path $Venv "Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu130
# Pin the CUDA torch just installed, otherwise pip may "upgrade" it to a CPU build from PyPI.
$torch = & $py -c "import torch; print(torch.__version__)"
& $py -m pip install -r "$PSScriptRoot\requirements.txt" "torch==$torch"
& $py -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
Write-Host "Next (optional, saves ~3 GB RAM at load): `$env:HF_HOME='$(Join-Path $Data 'models\huggingface')'; & '$py' convert_nllb.py facebook/nllb-200-distilled-1.3B '$(Join-Path $Data 'models\nllb-200-distilled-1.3B-fp16')'"
