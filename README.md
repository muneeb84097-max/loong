---
title: LongCat-Video-Avatar 1.5
emoji: 🎤
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
short_description: Audio-driven talking-head video generation (Meituan LongCat)
python_version: "3.10"
startup_duration_timeout: 1h
hardware: zero-a10g
suggested_hardware: zero-a10g
---

Audio-driven talking-head video generation using Meituan's LongCat-Video-Avatar 1.5.

Upload a reference image + audio + text prompt, get back a 5-second lip-synced video. Runs the
INT8-quantized DiT with the DMD2-distilled 8-step LoRA on ZeroGPU xlarge.

- Source: https://github.com/meituan-longcat/LongCat-Video
- Weights: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5
