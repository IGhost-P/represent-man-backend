#!/usr/bin/env bash
# exit on error
set -o errexit

# FFmpeg 설치 (가장 중요)
apt-get update && apt-get install -y ffmpeg

# 파이썬 라이브러리 설치
pip install -r requirements.txt
