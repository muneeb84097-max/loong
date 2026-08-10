"""Gradio ZeroGPU Space for LongCat-Video-Avatar 1.5 (single-person AI2V).

Follows the same pattern as multimodalart/LongCat-Video: download weights to
local container disk, eagerly construct the pipeline at module level (CPU dtype),
then `pipe.to(device)` once. Inside @spaces.GPU, spaces transparently materializes
weights on the real GPU.
"""

# spaces must be imported before torch
import spaces  # noqa: F401

import json
import hashlib
import math
import os
import shutil
import sys
import subprocess
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "weights"))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = WEIGHTS_DIR / "LongCat-Video"
AVATAR_DIR = WEIGHTS_DIR / "LongCat-Video-Avatar-1.5"
print(f"[boot] WEIGHTS_DIR={WEIGHTS_DIR.resolve()}", flush=True)

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import snapshot_download
import imageio
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------------------------------------------------------------------------
# 0) xformers → SDPA shim (the published xformers wheel is incompatible with
#    the torch version on zero-a10g; SDPA matches the call sites).
# ---------------------------------------------------------------------------

def _install_sdpa_shim():
    import xformers.ops

    class _BDShim:
        def __init__(self, q_seqlen, kv_seqlen):
            self.q_seqlen = list(q_seqlen)
            self.kv_seqlen = list(kv_seqlen)

        @classmethod
        def from_seqlens(cls, q_seqlen, kv_seqlen):
            return cls(q_seqlen, kv_seqlen)

    xformers.ops.fmha.attn_bias.BlockDiagonalMask = _BDShim

    def _meff(q, k, v, attn_bias=None, op=None, **_):
        if attn_bias is None:
            q_ = q.transpose(1, 2).contiguous()
            k_ = k.transpose(1, 2).contiguous()
            v_ = v.transpose(1, 2).contiguous()
            return F.scaled_dot_product_attention(q_, k_, v_).transpose(1, 2)
        if isinstance(attn_bias, _BDShim):
            outs, q_off, k_off = [], 0, 0
            for q_len, k_len in zip(attn_bias.q_seqlen, attn_bias.kv_seqlen):
                q_b = q[:, q_off:q_off + q_len].transpose(1, 2).contiguous()
                k_b = k[:, k_off:k_off + k_len].transpose(1, 2).contiguous()
                v_b = v[:, k_off:k_off + k_len].transpose(1, 2).contiguous()
                outs.append(F.scaled_dot_product_attention(q_b, k_b, v_b).transpose(1, 2))
                q_off += q_len
                k_off += k_len
            return torch.cat(outs, dim=1)
        raise NotImplementedError(f"Unsupported attn_bias in SDPA shim: {type(attn_bias)}")

    xformers.ops.memory_efficient_attention = _meff
    print("[boot] installed xformers→SDPA shim", flush=True)


_install_sdpa_shim()


# ---------------------------------------------------------------------------
# 1) Download weights (one-time per container) — local disk, no bucket
# ---------------------------------------------------------------------------

token = os.environ.get("HF_TOKEN")

if not (BASE_DIR / "vae" / "config.json").exists():
    print("[boot] downloading LongCat-Video (vae/text_encoder/tokenizer)…", flush=True)
    snapshot_download(
        "meituan-longcat/LongCat-Video",
        local_dir=str(BASE_DIR),
        token=token,
        allow_patterns=[
            "tokenizer/*",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ],
        ignore_patterns=[
            "text_encoder/*.fp32*",
            "text_encoder/*.bin",
            "text_encoder/flax_model*",
            "text_encoder/tf_model*",
            "vae/flax_model*",
            "vae/tf_model*",
        ],
    )

if not (AVATAR_DIR / "base_model_int8" / "config.json").exists():
    print("[boot] downloading LongCat-Video-Avatar-1.5 (INT8 + lora + whisper + vocal_separator)…", flush=True)
    snapshot_download(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        local_dir=str(AVATAR_DIR),
        token=token,
        allow_patterns=[
            "base_model_int8/*",
            "lora/*",
            "scheduler/*",
            "vocal_separator/*",
            "whisper-large-v3/model.safetensors",
            "whisper-large-v3/*.json",
            "whisper-large-v3/*.txt",
        ],
        ignore_patterns=[
            "whisper-large-v3/model.fp32*",
            "whisper-large-v3/flax_model*",
            "whisper-large-v3/tf_model*",
            "whisper-large-v3/pytorch_model*",
        ],
    )
print("[boot] weights ready", flush=True)


# ---------------------------------------------------------------------------
# 2) Patch DiT config so it uses xformers (== our SDPA shim) instead of flash.
# ---------------------------------------------------------------------------

_cfg_path = AVATAR_DIR / "base_model_int8" / "config.json"
if _cfg_path.exists():
    _cfg = json.loads(_cfg_path.read_text())
    _changed = False
    for k in ("enable_flashattn2", "enable_flashattn3", "enable_bsa"):
        if _cfg.get(k):
            _cfg[k] = False
            _changed = True
    if not _cfg.get("enable_xformers"):
        _cfg["enable_xformers"] = True
        _changed = True
    if _changed:
        _cfg_path.write_text(json.dumps(_cfg, indent=2))
        print("[boot] patched DiT config -> SDPA backend", flush=True)


# ---------------------------------------------------------------------------
# 3) Eager model load at module level (multimodalart/LongCat-Video pattern)
# ---------------------------------------------------------------------------

from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline  # noqa: E402
from longcat_video.modules.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402
from longcat_video.modules.quantization import load_quantized_dit  # noqa: E402
from longcat_video.audio_process import (  # noqa: E402
    get_audio_encoder,
    get_audio_feature_extractor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
CP_SPLIT_HW = [1, 1]

print(f"[boot] device={device} dtype={torch_dtype}", flush=True)
print("[boot] tokenizer + text_encoder…", flush=True); _t = time.time()
tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), subfolder="tokenizer", torch_dtype=torch_dtype)
text_encoder = UMT5EncoderModel.from_pretrained(str(BASE_DIR), subfolder="text_encoder", torch_dtype=torch_dtype)
print(f"[boot] text_encoder loaded in {time.time()-_t:.1f}s", flush=True)

print("[boot] VAE + scheduler…", flush=True); _t = time.time()
vae = AutoencoderKLWan.from_pretrained(str(BASE_DIR), subfolder="vae", torch_dtype=torch_dtype)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(AVATAR_DIR), subfolder="scheduler", torch_dtype=torch_dtype)
print(f"[boot] VAE+scheduler loaded in {time.time()-_t:.1f}s", flush=True)

print("[boot] INT8 DiT + DMD2 LoRA…", flush=True); _t = time.time()
dit = load_quantized_dit(str(AVATAR_DIR), subfolder="base_model_int8", cp_split_hw=CP_SPLIT_HW)
_lora_path = AVATAR_DIR / "lora" / "dmd_lora.safetensors"
if _lora_path.exists():
    dit.load_lora(str(_lora_path), "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    dit.enable_loras(["dmd"])
    print("[boot] DMD2 8-step LoRA enabled", flush=True)
print(f"[boot] DiT loaded in {time.time()-_t:.1f}s", flush=True)

print("[boot] Whisper-Large-v3…", flush=True); _t = time.time()
audio_encoder = get_audio_encoder(str(AVATAR_DIR / "whisper-large-v3"), "avatar-v1.5")
audio_feature_extractor = get_audio_feature_extractor(str(AVATAR_DIR / "whisper-large-v3"), "avatar-v1.5")
print(f"[boot] Whisper loaded in {time.time()-_t:.1f}s", flush=True)

print("[boot] vocal separator (Kim_Vocal_2)…", flush=True)
from audio_separator.separator import Separator  # noqa: E402

VOCAL_TMP = Path("/tmp/vocal_out")
(VOCAL_TMP / "vocals").mkdir(parents=True, exist_ok=True)
vocal_separator = Separator(
    output_dir=str(VOCAL_TMP / "vocals"),
    output_single_stem="vocals",
    model_file_dir=str(AVATAR_DIR / "vocal_separator"),
)
vocal_separator.load_model("Kim_Vocal_2.onnx")

print("[boot] assembling pipeline…", flush=True)
pipe = LongCatVideoAvatarPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    scheduler=scheduler,
    dit=dit,
    audio_encoder=audio_encoder,
    audio_feature_extractor=audio_feature_extractor,
    model_type="avatar-v1.5",
)
pipe.to(device)
# pipe.to() doesn't move the audio_encoder (Whisper); do it explicitly.
audio_encoder.to(device, dtype=torch_dtype)
print("[boot] ready.", flush=True)


# ---------------------------------------------------------------------------
# 4) Inference
# ---------------------------------------------------------------------------

NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

VOCAL_MODE_FAST = "Clean speech (fast)"
VOCAL_MODE_QUALITY = "Isolate vocals (quality)"
ACCEL_MODE_EXACT = "Exact 8-step"
ACCEL_MODE_DBCACHE = "DBCache fast"
ACCEL_MODE_DBCACHE_FASTER = "DBCache faster"
SAVE_FPS = 25
NUM_FRAMES = 125
VIDEO_SECONDS = NUM_FRAMES / SAVE_FPS
_AUDIO_EMB_CACHE = OrderedDict()
_VOCAL_CACHE = OrderedDict()
_CACHE_LIMIT = 8
_DISK_CACHE_DIR = Path(tempfile.gettempdir()) / "longcat_cache"
_AUDIO_CACHE_DIR = _DISK_CACHE_DIR / "audio_emb"
_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_CSS = """
main,
.gradio-container,
.fillable:not(.fill_width) {
  width: min(100%, 1320px) !important;
  max-width: 1320px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.prose blockquote {
  margin-top: 0 !important;
}
"""


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _cache_file(namespace: Path, key) -> Path:
    key_json = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return namespace / f"{hashlib.sha256(key_json.encode('utf-8')).hexdigest()}.pt"


def _load_audio_16k(path: str):
    try:
        import soundfile as sf
        from scipy.signal import resample_poly

        speech, sr = sf.read(path, dtype="float32", always_2d=False)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            gcd = math.gcd(int(sr), 16000)
            speech = resample_poly(speech, 16000 // gcd, int(sr) // gcd).astype(np.float32)
            sr = 16000
        return np.ascontiguousarray(speech, dtype=np.float32), sr
    except Exception as e:
        print(f"[audio] soundfile load failed, falling back to librosa: {e}", flush=True)
        import librosa

        speech, sr = librosa.load(path, sr=16000)
        return np.ascontiguousarray(speech, dtype=np.float32), sr


def _extract_vocal(src: str, audio_hash: str) -> str:
    cached_path = _cache_get(_VOCAL_CACHE, audio_hash)
    if cached_path and Path(cached_path).exists():
        print(f"[cache] vocal hit {audio_hash[:10]}", flush=True)
        return cached_path

    stable_path = VOCAL_TMP / "vocals" / f"{audio_hash[:16]}_vocals.wav"
    if stable_path.exists():
        _cache_put(_VOCAL_CACHE, audio_hash, str(stable_path))
        return str(stable_path)

    try:
        outputs = vocal_separator.separate(src)
        if outputs:
            separated = (VOCAL_TMP / "vocals" / outputs[0]).resolve()
            shutil.copyfile(separated, stable_path)
            _cache_put(_VOCAL_CACHE, audio_hash, str(stable_path))
            return str(stable_path)
    except Exception as e:
        print(f"[vocal] separation failed, using raw audio: {e}", flush=True)
    return src


def _check_duration(*args, **kwargs):
    return 240


def _prepare_audio_embedding(audio_path: str, vocal_mode: str, num_frames: int, save_fps: int, audio_stride: int, progress):
    audio_hash = _file_sha256(audio_path)
    cache_key = (audio_hash, vocal_mode, num_frames, save_fps, audio_stride)
    cached = _cache_get(_AUDIO_EMB_CACHE, cache_key)
    if cached is not None:
        progress(0.20, desc="Using cached audio conditioning…")
        print(f"[cache] audio embedding hit {audio_hash[:10]}", flush=True)
        return cached.to(device, non_blocking=True)
    cache_path = _cache_file(_AUDIO_CACHE_DIR, cache_key)
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu")
            _cache_put(_AUDIO_EMB_CACHE, cache_key, cached)
            progress(0.20, desc="Using cached audio conditioning…")
            print(f"[cache] audio embedding disk hit {audio_hash[:10]}", flush=True)
            return cached.to(device, non_blocking=True)
        except Exception as e:
            print(f"[cache] audio disk cache read failed: {e}", flush=True)

    t0 = time.perf_counter()
    if vocal_mode == VOCAL_MODE_QUALITY:
        progress(0.05, desc="Isolating vocals…")
        vocal_path = _extract_vocal(audio_path, audio_hash)
    else:
        progress(0.05, desc="Using clean speech directly…")
        vocal_path = audio_path
    print(f"[timing] audio_input_ready={time.perf_counter() - t0:.2f}s mode={vocal_mode} hash={audio_hash[:10]}", flush=True)

    t0 = time.perf_counter()
    speech, sr = _load_audio_16k(vocal_path)
    pad = math.ceil((num_frames / save_fps - len(speech) / sr) * sr)
    if pad > 0:
        speech = np.concatenate([speech, np.zeros(pad, dtype=speech.dtype)])
    print(f"[timing] audio_load={time.perf_counter() - t0:.2f}s sr={sr} samples={len(speech)}", flush=True)

    progress(0.15, desc="Encoding audio (Whisper-Large-v3)…")
    t0 = time.perf_counter()
    full_audio_emb = pipe.get_audio_embedding(
        speech, fps=save_fps * audio_stride, device=device, sample_rate=sr, model_type="avatar-v1.5"
    )
    if torch.isnan(full_audio_emb).any():
        raise gr.Error("Audio embedding contains NaN — try a different audio clip.")

    indices = torch.arange(2 * 2 + 1, device=full_audio_emb.device) - 2
    center = torch.arange(0, audio_stride * num_frames, audio_stride, device=full_audio_emb.device).unsqueeze(1) + indices.unsqueeze(0)
    center = torch.clamp(center, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center][None, ...].to(device)
    print(f"[timing] audio_encode={time.perf_counter() - t0:.2f}s shape={tuple(audio_emb.shape)}", flush=True)

    audio_emb_cpu = audio_emb.detach().cpu()
    _cache_put(_AUDIO_EMB_CACHE, cache_key, audio_emb_cpu)
    try:
        torch.save(audio_emb_cpu, cache_path)
    except Exception as e:
        print(f"[cache] audio disk cache write failed: {e}", flush=True)
    return audio_emb


def _save_video_ffmpeg_fast(frames: np.ndarray, out_base: Path, audio_path: str, fps: int, quality: int = 5) -> str:
    out_base = str(out_base)
    temp_video = out_base + "-video.mp4"
    out_path = out_base + ".mp4"

    writer = imageio.get_writer(temp_video, fps=fps, codec="libx264", quality=quality)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()

    duration = len(frames) / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        temp_video,
        "-i",
        audio_path,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    try:
        os.remove(temp_video)
    except OSError:
        pass
    return out_path


def _configure_dit_acceleration(acceleration: str):
    if acceleration in (ACCEL_MODE_DBCACHE, ACCEL_MODE_DBCACHE_FASTER):
        faster = acceleration == ACCEL_MODE_DBCACHE_FASTER
        pipe.dit.configure_dbcache(
            enabled=True,
            fn=1,
            bn=0,
            warmup_steps=1,
            max_cached_steps=3 if faster else 2,
            max_continuous_cached_steps=1,
            # The distilled 8-step schedule has larger residual deltas than
            # upstream 50-step DBCache examples, so cache only bounded steps.
            residual_diff_threshold=0.35,
            downsample_factor=4,
        )
        return "DMD2 8-step + DBCache" + (" faster" if faster else "")

    pipe.dit.configure_dbcache(enabled=False)
    return "DMD2 8-step"


@spaces.GPU(duration=_check_duration, size="xlarge")
def generate(
    image_path: str,
    audio_path: str,
    prompt: str,
    resolution: str,
    seed: int,
    vocal_mode: str = VOCAL_MODE_FAST,
    acceleration: str = ACCEL_MODE_DBCACHE_FASTER,
    progress=gr.Progress(track_tqdm=True),
):
    if not image_path:
        raise gr.Error("Please upload a reference image.")
    if not audio_path:
        raise gr.Error("Please upload an audio clip.")
    prompt = (prompt or "A person is talking naturally.").strip()

    save_fps = SAVE_FPS
    audio_stride = 1
    num_frames = NUM_FRAMES
    t_total = time.perf_counter()

    audio_emb = _prepare_audio_embedding(audio_path, vocal_mode, num_frames, save_fps, audio_stride, progress)

    generation_mode = _configure_dit_acceleration(acceleration)
    progress(0.30, desc=f"Generating video ({generation_mode})…")
    image = Image.open(image_path).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(int(seed))

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=8,
            text_guidance_scale=1.0,
            audio_guidance_scale=1.0,
            output_type="np",
            generator=generator,
            audio_emb=audio_emb,
            use_distill=True,
        )
    print(f"[timing] video_generate={time.perf_counter() - t0:.2f}s mode={acceleration}", flush=True)
    if acceleration in (ACCEL_MODE_DBCACHE, ACCEL_MODE_DBCACHE_FASTER):
        print(f"[dbcache] {pipe.dit.get_dbcache_stats()}", flush=True)

    progress(0.92, desc="Muxing audio + video…")
    t0 = time.perf_counter()
    frames = (output[0] * 255).astype(np.uint8)
    out_base = Path(tempfile.gettempdir()) / f"longcat_{uuid.uuid4().hex[:8]}"
    out_path = _save_video_ffmpeg_fast(frames, out_base, audio_path, fps=save_fps, quality=5)
    print(f"[timing] mux={time.perf_counter() - t0:.2f}s total={time.perf_counter() - t_total:.2f}s", flush=True)
    print(f"[gen] wrote {out_path}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# 5) Gradio UI
# ---------------------------------------------------------------------------

ASSET_DIR = Path(__file__).parent / "assets" / "avatar"
EXAMPLE_CACHE_VERSION = "elevenlabs-example-voices-v4"
EXAMPLES = []


def _reset_example_cache_if_needed():
    cache_root = Path(".gradio") / "cached_examples"
    marker = cache_root / ".longcat_example_version"
    try:
        current = marker.read_text().strip() if marker.exists() else ""
        if current != EXAMPLE_CACHE_VERSION:
            shutil.rmtree(cache_root, ignore_errors=True)
            cache_root.mkdir(parents=True, exist_ok=True)
            marker.write_text(EXAMPLE_CACHE_VERSION)
    except Exception as e:
        print(f"[cache] example cache reset skipped: {e}", flush=True)


def _add_example(image_path: Path, audio_path: Path, prompt_text: str, seed: int):
    if image_path.exists() and audio_path.exists():
        EXAMPLES.append([
            str(image_path),
            str(audio_path),
            prompt_text,
            "480p",
            seed,
            VOCAL_MODE_FAST,
            ACCEL_MODE_DBCACHE_FASTER,
        ])


_add_example(
    ASSET_DIR / "single" / "character.png",
    ASSET_DIR / "examples" / "audio" / "character_voice.wav",
    "A friendly cartoon character with short brown hair speaking to the camera "
    "against a soft blue background, simple flat illustration style.",
    42,
)
_add_example(
    ASSET_DIR / "examples" / "orc_warrior.png",
    ASSET_DIR / "examples" / "audio" / "orc_warrior_voice.wav",
    "A fantasy orc warrior portrait speaking directly to the camera, cinematic "
    "studio lighting, detailed armor, expressive face, smoky neutral backdrop.",
    77,
)
_add_example(
    ASSET_DIR / "examples" / "photoreal_person.png",
    ASSET_DIR / "examples" / "audio" / "photoreal_person_voice.wav",
    "A photorealistic person speaking naturally to the camera in a clean studio "
    "portrait, soft neutral background, realistic facial expression.",
    123,
)

_reset_example_cache_if_needed()

with gr.Blocks(title="LongCat-Video-Avatar 1.5", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        """
        # 🎤 LongCat-Video-Avatar 1.5: Audio-Image-to-Video

        Upload a reference image + audio clip + a short text prompt.
        Generates a ~5-second lip-synced video using Meituan's
        LongCat-Video-Avatar 1.5 (INT8 DiT + DMD2 8-step distilled).

        > 🤖 Built autonomously by an agent across real Spaces sessions. See [Building ZeroGPU Spaces Autonomously](https://huggingface.co/blog/victor/building-zerogpu-spaces-autonomously) for the story behind it.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Reference image", type="filepath")
            audio_in = gr.Audio(label="Driving audio", type="filepath")
            prompt = gr.Textbox(
                label="Prompt",
                value="A person is speaking expressively, looking at the camera.",
                lines=3,
            )
            with gr.Row():
                resolution = gr.Radio(["480p", "720p"], value="480p", label="Resolution")
                seed = gr.Number(value=42, precision=0, label="Seed")
            vocal_mode = gr.Radio(
                [VOCAL_MODE_FAST, VOCAL_MODE_QUALITY],
                value=VOCAL_MODE_FAST,
                label="Audio preprocessing",
            )
            acceleration = gr.Radio(
                [ACCEL_MODE_EXACT, ACCEL_MODE_DBCACHE, ACCEL_MODE_DBCACHE_FASTER],
                value=ACCEL_MODE_DBCACHE_FASTER,
                label="Acceleration",
            )
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            video_out = gr.Video(label="Output", autoplay=True, height=420)
            if EXAMPLES:
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[image_in, audio_in, prompt, resolution, seed, vocal_mode, acceleration],
                    outputs=video_out,
                    fn=generate,
                    cache_examples=True,
                    cache_mode="lazy",
                    examples_per_page=3,
                )

    go.click(
        generate,
        inputs=[image_in, audio_in, prompt, resolution, seed, vocal_mode, acceleration],
        outputs=video_out,
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch(show_error=True)
