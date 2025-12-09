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
import imageio, subprocess, os, inspect, sys, argparse
import numpy as np


def _get_device(stage=None):
    stage_key = f"MINI_SORA_{stage.upper()}_DEVICE" if stage else None
    stage_override = os.environ.get(stage_key) if stage_key else None
    if stage_override:
        return stage_override
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


def _even_size(val, minimum=2):
    """Clamp to even integer to satisfy 4:2:0 pixel formats."""
    if val is None:
        return None
    try:
        v = int(val)
    except (TypeError, ValueError):
        return None
    v = max(minimum, v)
    return v if v % 2 == 0 else v - 1


def _to_hwc_uint8(frame):
    """
    Normalize a single frame to HWC uint8 for imageio/ffmpeg.
    Handles torch tensors (TCHW/CHW), numpy arrays, and PIL images.
    """
    if torch is not None and isinstance(frame, torch.Tensor):
        f = frame.detach().cpu()
        if f.ndim == 4:
            # Take first batch dim if present
            f = f[0]
        if f.ndim == 3 and f.shape[0] in (1, 3, 4):
            f = f.permute(1, 2, 0)
        f = f.numpy()
    elif isinstance(frame, np.ndarray):
        f = frame
        if f.ndim == 3 and f.shape[0] in (1, 3, 4) and f.shape[0] != f.shape[-1]:
            f = np.transpose(f, (1, 2, 0))
    else:
        f = np.array(frame)
        if f.ndim == 3 and f.shape[0] in (1, 3, 4) and f.shape[0] != f.shape[-1]:
            f = np.transpose(f, (1, 2, 0))

    if f.ndim == 2:
        # Grayscale → RGB
        f = np.stack([f] * 3, axis=-1)
    f = np.clip(f, 0, 255)
    if f.dtype != np.uint8:
        # Assume input in [0,1] or float; scale if needed
        if f.max() <= 1.0:
            f = (f * 255.0)
        f = f.astype(np.uint8)
    return f


def _normalize_frames(frames):
    """
    Ensure frames is a list of HWC uint8 arrays.
    Supports torch tensors (TCHW / B T C H W), numpy arrays, or lists.
    """
    normalized = []

    def _add(item):
        # Convert torch tensors to numpy for easier handling
        if torch is not None and isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()

        if isinstance(item, np.ndarray):
            # If batched/stacked frames, split them
            if item.ndim >= 4 and item.shape[0] > 1:
                for i in range(item.shape[0]):
                    _add(item[i])
                return
            # If single frame with leading dim 1, squeeze it
            if item.ndim >= 4 and item.shape[0] == 1:
                item = np.squeeze(item, axis=0)
            normalized.append(_to_hwc_uint8(item))
            return

        if isinstance(item, (list, tuple)):
            for x in item:
                _add(x)
            return

        # Fallback: convert to numpy and handle potential stacked frames
        arr = np.array(item)
        if arr.ndim >= 4 and arr.shape[0] > 1:
            for i in range(arr.shape[0]):
                _add(arr[i])
            return
        if arr.ndim >= 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        normalized.append(_to_hwc_uint8(arr))

    _add(frames)
    return normalized


def _resize_frames(frames, target_w=None, target_h=None):
    if not target_w or not target_h:
        return frames
    tw, th = _even_size(target_w), _even_size(target_h)
    resized = []
    for f in frames:
        img = Image.fromarray(f)
        resized.append(np.array(img.resize((tw, th), Image.Resampling.LANCZOS)))
    return resized


def _resize_frames(frames, target_w=None, target_h=None):
    if not target_w or not target_h:
        return frames
    tw, th = _even_size(target_w), _even_size(target_h)
    resized = []
    for f in frames:
        img = Image.fromarray(f)
        resized.append(np.array(img.resize((tw, th), Image.Resampling.LANCZOS)))
    return resized


def _video_has_audio(path):
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def _probe_fps(path, fallback=None):
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ],
            capture_output=True, text=True, check=True
        )
        rate = res.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/")
            num, den = float(num), float(den)
            return num / den if den != 0 else None
        return float(rate)
    except Exception:
        return fallback


def _maybe_align_stage_devices(default_device):
    """
    If only one of the stage-specific device envs is set (and differs from default),
    ask whether to apply the same or default device to the other stage.
    Interactive only; skipped in test mode or non-tty.
    """
    if _is_test_mode() or not sys.stdin.isatty():
        return

    img_dev = os.environ.get("MINI_SORA_IMAGE_DEVICE")
    vid_dev = os.environ.get("MINI_SORA_VIDEO_DEVICE")

    def _prompt_set(missing_key, source_val):
        choice = input(
            f"You set {missing_key.replace('VIDEO', 'IMAGE').replace('IMAGE', 'VIDEO')}={source_val} "
            f"and left {missing_key} unset (default {default_device}). "
            f"Use the same device for {missing_key.split('_')[-2].lower()}? [y/N] "
        ).strip().lower()
        if choice.startswith("y"):
            os.environ[missing_key] = source_val
        else:
            os.environ[missing_key] = default_device

    if img_dev and not vid_dev and img_dev != default_device:
        _prompt_set("MINI_SORA_VIDEO_DEVICE", img_dev)
    elif vid_dev and not img_dev and vid_dev != default_device:
        _prompt_set("MINI_SORA_IMAGE_DEVICE", vid_dev)


def _parse_args():
    parser = argparse.ArgumentParser(description="Mini-Sora pipeline")
    parser.add_argument("--device", help="Global device override (cpu/mps/cuda)")
    parser.add_argument("--device-image", help="Image stage device (cpu/mps/cuda)")
    parser.add_argument("--device-video", help="Video stage device (cpu/mps/cuda)")
    parser.add_argument("--low-memory", dest="low_memory", action="store_true", help="Enable low-memory mode")
    parser.add_argument("--no-low-memory", dest="low_memory", action="store_false", help="Disable low-memory mode")
    parser.set_defaults(low_memory=None)
    parser.add_argument("--disable-safety", dest="disable_safety", action="store_true", help="Disable SD safety checker")
    parser.add_argument("--enable-safety", dest="disable_safety", action="store_false", help="Enable SD safety checker")
    parser.set_defaults(disable_safety=None)
    parser.add_argument("--sd-model", help="Stable Diffusion model id or path")
    parser.add_argument("--video-model", help="Video model id or path")
    parser.add_argument("--svd-width", type=int, help="SVD resolution width")
    parser.add_argument("--svd-height", type=int, help="SVD resolution height")
    parser.add_argument("--svd-frames", type=int, help="SVD frames")
    parser.add_argument("--svd-steps", type=int, help="SVD inference steps")
    parser.add_argument("--svd-fps", type=int, help="Output FPS for SVD video")
    parser.add_argument("--svd-decode-chunk", type=int, help="SVD decode chunk size")
    parser.add_argument("--rife-dir", help="Path to Practical-RIFE checkout")
    parser.add_argument("--rife-model", help="RIFE model folder (relative or absolute)")
    parser.add_argument("--refine-width", type=int, help="Refine output width")
    parser.add_argument("--refine-height", type=int, help="Refine output height")
    parser.add_argument("--test-mode", dest="test_mode", action="store_true", help="Enable MINI_SORA_TEST_MODE")
    parser.add_argument("--no-test-mode", dest="test_mode", action="store_false", help="Disable MINI_SORA_TEST_MODE")
    parser.set_defaults(test_mode=None)
    parser.add_argument("--interp-method", choices=["RIFE", "FILM", "NONE"], help="Frame interpolation choice")
    parser.add_argument("--audio-option", choices=["0", "1", "2", "3"], help="Audio selection: 0=None,1=Ambient,2=Music,3=Voice")
    parser.add_argument("--voice-text", help="Voice-over text when audio option is 3")
    parser.add_argument("--voice-lang", help="Voice-over language code (default en)")
    return parser.parse_args()


def _set_env(name, value):
    if value is not None:
        os.environ[name] = str(value)


def _apply_args_to_env(args):
    _set_env("MINI_SORA_DEVICE", args.device)
    _set_env("MINI_SORA_IMAGE_DEVICE", args.device_image)
    _set_env("MINI_SORA_VIDEO_DEVICE", args.device_video)
    if args.low_memory is not None:
        _set_env("MINI_SORA_LOW_MEMORY", "1" if args.low_memory else "0")
    if args.disable_safety is not None:
        _set_env("MINI_SORA_DISABLE_SAFETY", "1" if args.disable_safety else "0")
    if args.test_mode is not None:
        _set_env("MINI_SORA_TEST_MODE", "1" if args.test_mode else "0")
    _set_env("MINI_SORA_SD_MODEL", args.sd_model)
    _set_env("MINI_SORA_VIDEO_MODEL", args.video_model)
    _set_env("MINI_SORA_SVD_WIDTH", args.svd_width)
    _set_env("MINI_SORA_SVD_HEIGHT", args.svd_height)
    _set_env("MINI_SORA_SVD_FRAMES", args.svd_frames)
    _set_env("MINI_SORA_SVD_STEPS", args.svd_steps)
    _set_env("MINI_SORA_SVD_FPS", args.svd_fps)
    _set_env("MINI_SORA_SVD_DECODE_CHUNK", args.svd_decode_chunk)
    _set_env("MINI_SORA_RIFE_DIR", args.rife_dir)
    _set_env("MINI_SORA_RIFE_MODEL", args.rife_model)
    _set_env("MINI_SORA_REFINE_WIDTH", args.refine_width)
    _set_env("MINI_SORA_REFINE_HEIGHT", args.refine_height)
    _set_env("MINI_SORA_INTERP_METHOD", args.interp_method)
    _set_env("MINI_SORA_AUDIO_OPTION", args.audio_option)
    _set_env("MINI_SORA_VOICE_TEXT", args.voice_text)
    _set_env("MINI_SORA_VOICE_LANG", args.voice_lang)


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
    device = _get_device(stage="image")
    dtype = _get_dtype(device)
    model_id = os.environ.get("MINI_SORA_SD_MODEL", "runwayml/stable-diffusion-v1-5")
    local_only = _env_flag("HF_HUB_OFFLINE") or _env_flag("MINI_SORA_LOCAL_ONLY")
    try:
        pipe_img = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        if _env_flag("MINI_SORA_DISABLE_SAFETY") and hasattr(pipe_img, "safety_checker"):
            pipe_img.safety_checker = None
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
    # Resize to requested dimensions if provided (default to SVD dims when set)
    img_w = _even_size(_env_int("MINI_SORA_IMG_WIDTH", _env_int("MINI_SORA_SVD_WIDTH", None)))
    img_h = _even_size(_env_int("MINI_SORA_IMG_HEIGHT", _env_int("MINI_SORA_SVD_HEIGHT", None)))
    if img_w and img_h:
        image = image.resize((img_w, img_h))
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
    device = _get_device(stage="video")
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
    base_image = Image.open(init_image_path).convert("RGB")
    base_w, base_h = base_image.size

    # Respect user-provided dims; only shrink for low_mem if not provided
    user_w = _env_int("MINI_SORA_SVD_WIDTH", None)
    user_h = _env_int("MINI_SORA_SVD_HEIGHT", None)

    def _calc_dims(w, h):
        if w and h:
            return _even_size(w), _even_size(h)
        if w and not h:
            h_est = int(round((w / base_w) * base_h)) if base_w else w
            return _even_size(w), _even_size(h_est)
        if h and not w:
            w_est = int(round((h / base_h) * base_w)) if base_h else h
            return _even_size(w_est), _even_size(h)
        # No user dims; start from base image size
        return _even_size(base_w), _even_size(base_h)

    width, height = _calc_dims(user_w, user_h)
    if not user_w and not user_h:
        # Fallback defaults when nothing provided
        default_w = 1024 if not low_mem else 512
        default_h = 576 if not low_mem else 288
        width = width or _even_size(default_w)
        height = height or _even_size(default_h)

    if low_mem:
        width = _even_size(int(width * 0.5))
        height = _even_size(int(height * 0.5))

    init_image = base_image.resize((width, height))

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
    try:
        frames = _normalize_frames(frames)
    except Exception as exc:
        print(f"⚠️ Failed to normalize frames: type={type(frames)}, err={exc}")
        raise
    frames = _resize_frames(frames, width, height)
    # Debug: print basic frame info to help diagnose channel/layout issues
    if frames:
        sample = frames[0]
        print(f"ℹ️ Video frames normalized: count={len(frames)}, shape={getattr(sample, 'shape', None)}, dtype={getattr(sample, 'dtype', None)}")
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
    input_video_abs = os.path.abspath(input_video)
    output_video_abs = os.path.abspath(output_video)
    _ensure_parent(output_video_abs)
    # Use detected FPS to avoid RIFE attempting audio merge (we set fps explicitly)
    fps_in = _probe_fps(input_video_abs)
    if fps_in is None:
        try:
            fps_in = float(os.environ.get("MINI_SORA_SVD_FPS", 7))
        except (TypeError, ValueError):
            fps_in = 7.0
    fps_out = max(1, int(round(fps_in * 2)))  # exp=1 doubles frames
    if method.upper() == "RIFE":
        rife_model = os.environ.get("MINI_SORA_RIFE_MODEL", "")
        rife_dir = os.environ.get(
            "MINI_SORA_RIFE_DIR",
            os.path.join(os.path.dirname(__file__), "practical_rife")
        )
        if not os.path.isdir(rife_dir):
            raise RuntimeError(
                f"RIFE directory not found at '{rife_dir}'. "
                "Set MINI_SORA_RIFE_DIR to your RIFE checkout (PRACTICAL-RIFE)."
            )
        # If the model path is not absolute, assume train_log/<model> under rife_dir
        if rife_model:
            if os.path.isabs(rife_model):
                model_path = rife_model
            else:
                model_path = os.path.join(rife_dir, "train_log", rife_model)
        else:
            # Default to train_log root
            model_path = os.path.join(rife_dir, "train_log")
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"RIFE model path not found: {model_path}. "
                "Place the weights under practical_rife/train_log/ (or a subfolder) or set "
                "MINI_SORA_RIFE_MODEL to an absolute path."
            )
        cmd = ["python", "inference_video.py", "--exp", "1",
               "--model", model_path,
               "--video", input_video_abs, "--output", output_video_abs,
               "--fps", str(fps_out)]
        subprocess.run(cmd, check=True, cwd=rife_dir)
        return output_video_abs
    elif method.upper() == "FILM":
        cmd = ["python", "-m", "film.interpolate",
               "--input_video", input_video_abs,
               "--output_video", output_video_abs,
               "--times_to_interpolate", "1"]
    else:
        raise ValueError("Interpolation method must be 'RIFE' or 'FILM'.")
    subprocess.run(cmd, check=True)
    return output_video_abs


def refine_video(input_video, output_video="refined.mp4"):
    if _is_test_mode():
        _ensure_parent(output_video)
        with open(output_video, "wb"):
            pass
        return output_video

    # Use SVD dimensions by default to keep stages aligned; optional refine override
    target_w = _even_size(_env_int("MINI_SORA_REFINE_WIDTH", _env_int("MINI_SORA_SVD_WIDTH", 1920)))
    target_h = _even_size(_env_int("MINI_SORA_REFINE_HEIGHT", _env_int("MINI_SORA_SVD_HEIGHT", 1080)))
    scale_filter = f"scale={target_w}:{target_h}:flags=lanczos"

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"{scale_filter},"
               "eq=contrast=1.05:brightness=0.02:saturation=1.1",
        "-c:v", "hevc_videotoolbox", "-q:v", "50", "-pix_fmt", "yuv420p", #"-preset", "medium",
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
    has_audio = _video_has_audio(video_path)
    if has_audio:
        filter_complex = "[1:a]volume=1.0[a1];[0:a][a1]amix=inputs=2:duration=shortest[a]"
        map_audio = "[a]"
    else:
        # No audio in video; just apply volume to the external track
        filter_complex = "[1:a]volume=1.0[a]"
        map_audio = "[a]"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", resolved_audio,
        "-shortest",
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", map_audio,
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
    args = _parse_args()
    _apply_args_to_env(args)
    default_dev = _get_device()
    _maybe_align_stage_devices(default_dev)
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

    # Interpolation choice: env/arg override, else prompt
    interp_method = os.environ.get("MINI_SORA_INTERP_METHOD", "").upper()
    if interp_method not in ["RIFE", "FILM", "NONE"]:
        interp_method = input("Choose interpolation method (RIFE / FILM / none): ").strip().upper()
    if interp_method in ["RIFE", "FILM"]:
        vid_path = interpolate_frames(vid_path, f"outputs/interpolated_{interp_method}.mp4", interp_method)

    refined_path = refine_video(vid_path, "outputs/refined.mp4")

    print("\nAudio options:\n 1 = Ambient\n 2 = Music\n 3 = Auto Voice-over (gTTS)\n 0 = None")
    choice = os.environ.get("MINI_SORA_AUDIO_OPTION", "").strip()
    if choice not in {"0", "1", "2", "3"}:
        choice = input("Select audio option: ").strip()
    audio_map = {"1": "audio/ambient.wav", "2": "audio/music.mp3"}

    final_path = refined_path
    if choice == "3":
        narration = os.environ.get("MINI_SORA_VOICE_TEXT", "").strip()
        if not narration:
            narration = input("Enter your voice-over text: ").strip() or \
                        "A calm morning by the lake. She splashes her face and walks toward the shore."
        lang = os.environ.get("MINI_SORA_VOICE_LANG", "").strip() or "en"
        voice_path = generate_voiceover(narration, lang, "audio/voice.wav")
        final_path = add_audio_to_video(refined_path, voice_path, "outputs/final_with_voice.mp4")
    elif choice in audio_map:
        final_path = add_audio_to_video(refined_path, audio_map[choice], "outputs/final_with_audio.mp4")

    print(f"\n🎬 Done! Final video saved as: {final_path}")
