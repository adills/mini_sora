# Mini-Sora main pipeline script (text→video→audio)

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import DiffusionPipeline
except ImportError:
    DiffusionPipeline = None
from PIL import Image
from gtts import gTTS
import imageio, subprocess, os, inspect


def _get_device():
    override = os.environ.get("MINI_SORA_DEVICE")
    if override:
        return override
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_dtype(device):
    if torch is None:
        return None
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def _env_flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _is_test_mode():
    return _env_flag("MINI_SORA_TEST_MODE")


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def generate_image(prompt, output_path="frame0.png"):
    if _is_test_mode():
        _ensure_parent(output_path)
        Image.new("RGB", (4, 4), (0, 0, 0)).save(output_path)
        return output_path
    if torch is None or DiffusionPipeline is None:
        raise ImportError(
            "generate_image requires torch and diffusers. "
            "Install the dependencies listed in Pipfile/README."
        )
    print("🎨 Generating initial image...")
    device = _get_device()
    dtype = _get_dtype(device)
    model_id = os.environ.get("MINI_SORA_SD_MODEL", "runwayml/stable-diffusion-v1-5")
    local_only = _env_flag("HF_HUB_OFFLINE") or _env_flag("MINI_SORA_LOCAL_ONLY")
    try:
        pipe_img = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load image model '{model_id}'. "
            "Download it once (huggingface-cli download) or set MINI_SORA_SD_MODEL to a local path. "
            "Set MINI_SORA_TEST_MODE=1 for stubbed outputs."
        ) from exc
    pipe_img.to(device)
    # Recommended if your computer has < 64 GB of RAM
    pipe_img.enable_attention_slicing()
    image = pipe_img(prompt, guidance_scale=8.0).images[0]
    image.save(output_path)
    return output_path


def generate_video(init_image_path, prompt, output_video="raw_output.mp4"):
    if _is_test_mode():
        _ensure_parent(output_video)
        with open(output_video, "wb"):
            pass
        return output_video
    if torch is None or DiffusionPipeline is None:
        raise ImportError(
            "generate_video requires torch and diffusers. "
            "Install the dependencies listed in Pipfile/README."
        )
    print("🎥 Generating motion video...")
    device = _get_device()
    dtype = _get_dtype(device)
    # Prefer new env var, but keep backward compatibility with the old name
    model_id = (
        os.environ.get("MINI_SORA_VIDEO_MODEL")
        or os.environ.get("MINI_SORA_WAVER_MODEL")
        or "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    )
    local_only = _env_flag("HF_HUB_OFFLINE") or _env_flag("MINI_SORA_LOCAL_ONLY")
    try:
        pipe_vid = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_only,
            variant="fp16" if dtype == torch.float16 else None
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load video model '{model_id}'. "
            "If you are offline or the model requires auth, download once with "
            "`hf download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir ./models/svd` "
            "and set MINI_SORA_VIDEO_MODEL=./models/svd. "
            "Set MINI_SORA_TEST_MODE=1 for stubbed outputs."
        ) from exc
    pipe_vid.to(device)
    pipe_vid.enable_attention_slicing()
    low_mem = _env_flag("MINI_SORA_LOW_MEMORY") or device == "cpu"
    if low_mem:
        print("⚠️ Low-memory video mode: reducing resolution/frames.")
    width = _env_int("MINI_SORA_SVD_WIDTH", 1024 if not low_mem else 512)
    height = _env_int("MINI_SORA_SVD_HEIGHT", 576 if not low_mem else 288)
    init_image = Image.open(init_image_path).convert("RGB").resize((width, height))

    # Lighten memory pressure when supported
    if hasattr(pipe_vid, "enable_vae_slicing"):
        pipe_vid.enable_vae_slicing()
    if hasattr(pipe_vid, "enable_vae_tiling") and low_mem:
        pipe_vid.enable_vae_tiling()
    if hasattr(pipe_vid, "enable_model_cpu_offload") and low_mem and device != "cpu":
        pipe_vid.enable_model_cpu_offload()

    # Detect supported args for the loaded pipeline (SVD doesn't take `prompt`)
    sig_params = set(inspect.signature(pipe_vid.__call__).parameters.keys())
    call_kwargs = {"image": init_image}
    if "prompt" in sig_params:
        call_kwargs["prompt"] = prompt
    else:
        if prompt:
            print("ℹ️ Video model ignores text prompts; motion is derived from the input image.")

    # Default knobs for Stable Video Diffusion (SVD)
    svd_frames = _env_int("MINI_SORA_SVD_FRAMES", 12 if not low_mem else 6)
    svd_steps = _env_int("MINI_SORA_SVD_STEPS", 32 if not low_mem else 16)
    svd_min_guidance = _env_float("MINI_SORA_SVD_MIN_GUIDE", 1.0)
    svd_max_guidance = _env_float("MINI_SORA_SVD_MAX_GUIDE", 3.0)
    svd_motion_bucket = _env_int("MINI_SORA_SVD_MOTION_BUCKET", 127)
    svd_noise_aug = _env_float("MINI_SORA_SVD_NOISE_AUG", 0.02)
    svd_decode_chunk = _env_int("MINI_SORA_SVD_DECODE_CHUNK", 6 if not low_mem else 3)
    svd_fps = _env_int("MINI_SORA_SVD_FPS", 7 if not low_mem else 6)

    # Backward-compatible knobs for Waver-style pipelines
    waver_frames = int(os.environ.get("MINI_SORA_WAVER_FRAMES", 48))
    waver_guidance = float(os.environ.get("MINI_SORA_WAVER_GUIDANCE", 8.0))
    waver_motion = float(os.environ.get("MINI_SORA_WAVER_MOTION", 0.7))

    def _maybe_set(name, value):
        if name in sig_params and value is not None:
            call_kwargs[name] = value

    # Prefer SVD-style params; fall back to Waver-style if supported
    _maybe_set("num_frames", svd_frames)
    _maybe_set("num_inference_steps", svd_steps)
    _maybe_set("min_guidance_scale", svd_min_guidance)
    _maybe_set("max_guidance_scale", svd_max_guidance)
    _maybe_set("motion_bucket_id", svd_motion_bucket)
    _maybe_set("noise_aug_strength", svd_noise_aug)
    _maybe_set("decode_chunk_size", svd_decode_chunk)
    _maybe_set("fps", svd_fps)

    _maybe_set("guidance_scale", waver_guidance)
    _maybe_set("motion_strength", waver_motion)

    # If the pipeline expects a different frame count (e.g., Waver), override
    if "motion_strength" in sig_params and "num_frames" in sig_params:
        call_kwargs["num_frames"] = waver_frames

    result = pipe_vid(**call_kwargs)
    frames = getattr(result, "frames", None) if result is not None else None
    if frames is None and isinstance(result, dict):
        frames = result.get("frames")
    if frames is None:
        frames = result
    fps_to_save = getattr(result, "fps", None) or call_kwargs.get("fps", svd_fps)
    try:
        imageio.mimsave(output_video, frames, fps=fps_to_save)
    except ValueError as exc:
        raise RuntimeError(
            "Failed to save video. Install ffmpeg support with "
            "`pip install \"imageio[ffmpeg]\"` (or `pip install imageio-ffmpeg`) "
            "and ensure system ffmpeg is available."
        ) from exc
    return output_video


def interpolate_frames(input_video, output_video="interpolated.mp4", method="RIFE"):
    print(f"🌀 Interpolating frames using {method}...")
    if method.upper() == "RIFE":
        cmd = ["python", "-m", "rife.infer", "--exp", "1",
               "--video", input_video, "--output", output_video]
    elif method.upper() == "FILM":
        cmd = ["python", "-m", "film.interpolate",
               "--input_video", input_video,
               "--output_video", output_video,
               "--times_to_interpolate", "1"]
    else:
        raise ValueError("Interpolation method must be 'RIFE' or 'FILM'.")
    subprocess.run(cmd, check=True)
    return output_video


def refine_video(input_video, output_video="refined.mp4"):
    if _is_test_mode():
        _ensure_parent(output_video)
        with open(output_video, "wb"):
            pass
        return output_video
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", "scale=1920:1080:flags=lanczos,"
               "eq=contrast=1.05:brightness=0.02:saturation=1.1",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        output_video
    ]
    subprocess.run(cmd, check=True)
    return output_video


def add_audio_to_video(video_path, audio_choice, output_path="final_with_audio.mp4"):
    if _is_test_mode():
        _ensure_parent(output_path)
        with open(output_path, "wb"):
            pass
        return output_path
    resolved_audio = audio_choice
    if not os.path.exists(resolved_audio):
        # If a .wav is missing, fall back to a sibling .mp3 (and vice versa)
        base, ext = os.path.splitext(audio_choice)
        alt_ext = ".mp3" if ext.lower() == ".wav" else ".wav"
        alt_path = f"{base}{alt_ext}"
        if os.path.exists(alt_path):
            resolved_audio = alt_path
        else:
            print(f"⚠️ Audio file not found: {audio_choice}")
            return video_path
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", resolved_audio,
        "-shortest",
        "-filter_complex",
        "[1:a]volume=1.0[a1];[0:a][a1]amix=inputs=2:duration=shortest[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def generate_voiceover(text, lang="en", output_file="audio/voice.wav"):
    if _is_test_mode():
        _ensure_parent(output_file)
        with open(output_file, "wb"):
            pass
        return output_file
    tts = gTTS(text=text, lang=lang)
    _ensure_parent(output_file)
    tts.save(output_file)
    return output_file


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    text_prompt = (
        "A serene woman standing ankle-deep in a calm lake at sunrise, "
        "soft golden lighting, cinematic composition, realistic detail."
    )
    motion_prompt = (
        "The woman bends down to splash water on her face, "
        "then turns and walks slowly toward the camera. "
        "Cinematic tone, morning light reflection, slow motion."
    )

    img_path = generate_image(text_prompt, "outputs/frame0.png")
    vid_path = generate_video(img_path, motion_prompt, "outputs/raw_output.mp4")

    interp_method = input("Choose interpolation method (RIFE / FILM / none): ").strip().upper()
    if interp_method in ["RIFE", "FILM"]:
        vid_path = interpolate_frames(vid_path, f"outputs/interpolated_{interp_method}.mp4", interp_method)

    refined_path = refine_video(vid_path, "outputs/refined.mp4")

    print("\nAudio options:\n 1 = Ambient\n 2 = Music\n 3 = Auto Voice-over (gTTS)\n 0 = None")
    choice = input("Select audio option: ").strip()
    audio_map = {"1": "audio/ambient.wav", "2": "audio/music.mp3"}

    final_path = refined_path
    if choice == "3":
        narration = input("Enter your voice-over text: ").strip() or \
                    "A calm morning by the lake. She splashes her face and walks toward the shore."
        lang = input("Enter language code (default 'en'): ").strip() or "en"
        voice_path = generate_voiceover(narration, lang, "audio/voice.wav")
        final_path = add_audio_to_video(refined_path, voice_path, "outputs/final_with_voice.mp4")
    elif choice in audio_map:
        final_path = add_audio_to_video(refined_path, audio_map[choice], "outputs/final_with_audio.mp4")

    print(f"\n🎬 Done! Final video saved as: {final_path}")
