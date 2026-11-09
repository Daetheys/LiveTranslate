#!/usr/bin/env bash
# macOS / Linux launcher. On macOS the terminal app needs microphone permission
# (System Settings > Privacy & Security > Microphone) the first time.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python app.py --log-dir "${LIVETRANSLATE_LOGS:-$HOME/LiveTranslate/logs}" "$@"
