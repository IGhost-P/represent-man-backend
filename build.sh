#!/usr/bin/env bash
# exit on error
set -o errexit

# FFmpeg 및 PyAudio에 필요한 시스템 라이브러리 설치
apt-get update && apt-get install -y ffmpeg portaudio19-dev

# 파이썬 라이브러리 설치
pip install -r requirements.txt
