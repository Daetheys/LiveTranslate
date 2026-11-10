# Starts the desktop app (Windows). Use run.cmd to double-click it: Windows blocks .ps1 files by default.
# Models, HF cache and session logs all live under $env:LIVETRANSLATE_DATA (default D:\).
$Data = if ($env:LIVETRANSLATE_DATA) { $env:LIVETRANSLATE_DATA } else { "D:\" }
$env:HF_HOME = Join-Path $Data "models\huggingface"

# Once all default models are downloaded, don't contact the Hub at startup.
$cached = "openai--whisper-large-v3-turbo", "Helsinki-NLP--opus-mt-fr-hu", "Helsinki-NLP--opus-mt-hu-fr",
          "csukuangfj--speaker-embedding-models" |
          ForEach-Object { Test-Path (Join-Path $env:HF_HOME "hub\models--$_") }
if ($cached -notcontains $false) { $env:HF_HUB_OFFLINE = "1" }

# NLLB is only loaded for languages other than FR/HU. Use the fp16 safetensors copy if it exists
# (see convert_nllb.py): the original 5.5 GB fp32 .bin needs ~8 GB of free RAM + page file to load.
$extra = @()
$nllb = Join-Path $Data "models\nllb-200-distilled-1.3B-fp16"
if (Test-Path (Join-Path $nllb "model.safetensors")) { $extra += @("--nllb-model", $nllb) }

& (Join-Path $Data "venvs\livetranslate\Scripts\python.exe") "$PSScriptRoot\app.py" `
    --log-dir (Join-Path $Data "LiveTranslate\logs") @extra @args
