# Starts the desktop app (Windows). Use run.cmd to double-click it: Windows blocks .ps1 files by default.
# Models, HF cache and session logs all live under $env:LIVETRANSLATE_DATA (default D:\).
$Data = if ($env:LIVETRANSLATE_DATA) { $env:LIVETRANSLATE_DATA } else { "D:\" }
$env:HF_HOME = Join-Path $Data "models\huggingface"

# Once the models are downloaded, don't contact the Hub at startup.
if (Test-Path (Join-Path $env:HF_HOME "hub\models--openai--whisper-large-v3-turbo")) { $env:HF_HUB_OFFLINE = "1" }

# NLLB converted to fp16 safetensors (see convert_nllb.py): the original 5.5 GB fp32 .bin needs
# ~8 GB of free RAM + page file to load and crashed on the dev machine.
$extra = @()
$nllb = Join-Path $Data "models\nllb-200-distilled-1.3B-fp16"
if (Test-Path (Join-Path $nllb "model.safetensors")) { $extra += @("--nllb-model", $nllb) }

& (Join-Path $Data "venvs\livetranslate\Scripts\python.exe") "$PSScriptRoot\app.py" `
    --log-dir (Join-Path $Data "LiveTranslate\logs") @extra @args
