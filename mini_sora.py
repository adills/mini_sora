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
import imageio, subprocess, os


def _get_device():
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


def _is_test_mode():
    return os.environ.get("MINI_SORA_TEST_MODE", "").lower() in {"1", "true", "yes", "on"}


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
    pipe_img = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    )
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
    pipe_vid = DiffusionPipeline.from_pretrained(
        "FoundationVision/Waver-I2V",
        torch_dtype=dtype,
        trust_remote_code=True
    )
    pipe_vid.to(device)
    pipe_vid.enable_attention_slicing()
    init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))
    result = pipe_vid(image=init_image, prompt=prompt, num_frames=48,
                      guidance_scale=8.0, motion_strength=0.7)
    imageio.mimsave(output_video, result.frames, fps=24)
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
