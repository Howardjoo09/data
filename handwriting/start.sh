#!/usr/bin/env sh
# 손글씨 노트 변환기 실행 (macOS / Linux)
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv || { echo "Python 3.10 이상이 필요합니다."; exit 1; }
fi
.venv/bin/python -m pip install -q -r requirements.txt || exit 1
exec .venv/bin/python server.py --open
