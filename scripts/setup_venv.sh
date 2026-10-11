#!/usr/bin/env bash
# Erstellt .venv und installiert alle Pakete (macOS / Linux).
#   bash scripts/setup_venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10)' || {
  echo "Python 3.10 oder neuer nötig (gefunden: $("$PY" --version 2>&1))" >&2; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -c "import torch; d='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'; print(f'OK: torch {torch.__version__}, device = {d}')"
echo "Aktivieren mit:  source .venv/bin/activate"
