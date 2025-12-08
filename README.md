# 🎬 Mini-Sora Local AI Video Generation Pipeline  
**Author:** Your Name  
**Version:** 1.0  
**Platform:** macOS (Apple Silicon M1/M2) or Linux (CUDA GPU)  
**Python:** 3.10+

---

## 🧩 Overview

**Mini-Sora** is an open-source, fully local text-to-video generation pipeline inspired by OpenAI’s Sora and InVideo AI.  
It produces short cinematic clips by chaining together:

1. **Text → Image** (Stable Diffusion)  
2. **Image → Video** (Stable Video Diffusion)  
3. **Frame Interpolation** (RIFE or FILM)  
4. **Video Refinement** (ffmpeg color grading + upscaling)  
5. **Audio Integration** (ambient, music, or auto-generated voice-over via gTTS)

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|------------|----------|--------------|
| macOS | 13.0+ | M1/M2 Pro/Max |
| GPU | Apple GPU / NVIDIA 3060+ | NVIDIA 4090 or M2 Max |
| Memory | 16 GB | 32+ GB |
| Disk Space | 20 GB free | 50 GB |
| Python | 3.10+ | 3.11 |

---

## 🧠 Dependencies

Install via **venv** or **Pipenv**:

```bash
git clone https://github.com/adills/mini_sora.git
cd mini_sora
python3 -m venv waver_env
source waver_env/bin/activate
# NOTE: On Mac with MPS, you don't need to specify the index-url in the next line
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate pillow "imageio[ffmpeg]" gTTS # imageio-ffmpeg for mp4 writing
pip install pytest
brew install ffmpeg

# RIFE
git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife && pip install -r requirements.txt

# FILM
pip install film
```

**Pipenv**
```bash
git clone https://github.com/adills/mini_sora.git
cd mini_sora
pip install pipenv
pipenv --python 3.12
pipenv shell
# NOTE: On Mac with MPS, you don't need to specify the index-url in the next line
pipenv install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pipenv install diffusers transformers accelerate pillow "imageio[ffmpeg]" gTTS # imageio-ffmpeg for mp4 writing
pipenv install --dev pytest
brew install ffmpeg

# RIFE
git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife && pip install -r requirements.txt

# FILM
pipenv install film
```

### Optional (for interpretation)
#### RIFE
git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife && pip install -r requirements.txt

#### FILM
pip install film

## 📦 Model downloads / offline use
- First run will download `runwayml/stable-diffusion-v1-5` (text→image) and `stabilityai/stable-video-diffusion-img2vid-xt-1-1` (image→video). If either is gated, run `hf auth login` (uses a token from https://huggingface.co/settings/tokens) and accept the license.
- To run fully offline, download once and point the env vars to the local folders:
  - `hf download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir ./models/svd`
  - `export MINI_SORA_VIDEO_MODEL=./models/svd` (old name `MINI_SORA_WAVER_MODEL` still works)
  - (optional) `export MINI_SORA_SD_MODEL=./models/stable-diffusion-v1-5`
- For a quick smoke test without downloads, set `MINI_SORA_TEST_MODE=1` to stub out the heavy stages.

## Usage
Run tests:
```bash
pytest -s tests/test_pipeline_e2e.py
```
The -s flag lets you see printed status lines such as:
```bash
🎨 Generating initial image...
🎥 Generating motion video...
✅ E2E voice-over test completed.
Final output: /tmp/pytest-.../final_with_voice.mp4
```

Run the main pipeline interactively:

```bash
python mini_sora.py
```

### Input during run
You’ll be prompted to:
	1.	Choose an interpolation method (RIFE / FILM / none).
	2.	Select audio option: Ambient / Music / Auto Voice-over / None.
	3.	Optionally enter voice-over text and language code.

```bash
Choose interpolation method (RIFE / FILM / none): none
Audio options:
  1 = Ambient
  2 = Music
  3 = Auto Voice-over (gTTS)
  0 = None
Select audio option: 3
Enter your voice-over text (or press Enter for default): A peaceful morning by the lake.
Enter voice language code (default 'en'): en
```

Note: Stable Video Diffusion is image-conditioned, so the “motion prompt” text is ignored in the current default video model.

### Memory tips
- If you hit out-of-memory or large buffer errors, try `MINI_SORA_LOW_MEMORY=1` (uses smaller resolution/frames) or override `MINI_SORA_SVD_FRAMES=6 MINI_SORA_SVD_WIDTH=512 MINI_SORA_SVD_HEIGHT=288`.
- Lower decode chunking if needed: `MINI_SORA_SVD_DECODE_CHUNK=3`.
- To force CPU instead of MPS/GPU (very slow, but safer for memory): `MINI_SORA_DEVICE=cpu`.
- To bypass the Stable Diffusion safety checker (e.g., if you keep getting black images), set `MINI_SORA_DISABLE_SAFETY=1`.

**Example Mac MPS minimum low memory CLI**
```bash
MINI_SORA_DEVICE=mps \
MINI_SORA_LOW_MEMORY=1 \
MINI_SORA_DISABLE_SAFETY=1 \
MINI_SORA_SVD_FRAMES=4 \
MINI_SORA_SVD_STEPS=8 \
MINI_SORA_SVD_WIDTH=320 \
MINI_SORA_SVD_HEIGHT=180 \
MINI_SORA_SVD_DECODE_CHUNK=1 \
python3 mini_sora.py
```

**Example of a mixed MPS and CPU process**
```bash
MINI_SORA_IMAGE_DEVICE=mps \
MINI_SORA_VIDEO_DEVICE=cpu \
MINI_SORA_LOW_MEMORY=1 \
MINI_SORA_DISABLE_SAFETY=1 \
MINI_SORA_SVD_FRAMES=8 \
MINI_SORA_SVD_STEPS=16 \
MINI_SORA_SVD_WIDTH=256 \
MINI_SORA_SVD_HEIGHT=448 \
MINI_SORA_SVD_DECODE_CHUNK=3 \
python3 mini_sora.py
```

### Output
✅ Voice-over saved: audio/voice.wav
✅ Audio-integrated video ready: outputs/final_with_voice.mp4
🎬 Done! Final video saved as: outputs/final_with_voice.mp4

The final video is saved under:
```outputs/final_with_audio.mp4```

## 🎙️ Supported Voice Languages (gTTS)
| Code  | Description |
|:-----:|:------------|
| en    | English (US) |
| en-uk | English (UK) |
| en-au | English (Australia) |
| fr    | French |
| es    | Spanish |
| ja    | Japanese |
| hi    | Hindi |
| zh-cn | Chinese (Simplified) |


## 🧩 Extensible Modules
| Stage         | Function                                    | File         |
|:--------------|:--------------------------------------------|:-------------|
| Text → Image  | generate_image()                            | mini_sora.py |
| Image → Video | generate_video()                            | mini_sora.py |
| Interpolation | interpolate_frames()                        | mini_sora.py |
| Refinement    | refine_video()                              | mini_sora.py |
| Audio / Voice | add_audio_to_video() / generate_voiceover() | mini_sora.py |

## 🧠 Notes for Developers
- Designed for modular import into a Django/Flask backend if needed.
- Each stage returns a file path and can be orchestrated via an external API.
- You can disable any module via flags in the main workflow.

## ✅ Future Enhancements
- Bark / Coqui-TTS integration for offline neural voice synthesis
- Audio beat synchronization using librosa
- Video stabilization for handheld-like motion
- REST API layer for external orchestration

## 🧩 License
MIT License © 2025 — Attribution required.

## 🧪 Example Output
> A 5-second cinematic clip of a woman by a lake, generated fully on-device.
> Output file: outputs/final_with_voice.mp4

## File structure
```text
mini-sora/
├── mini_sora.py                # main script
├── tests/
│   ├── test_pipeline_unit.py   # unit tests
│   └── test_pipeline_e2e.py    # full end-to-end test
├── outputs/                    # generated images/videos
├── audio/
│   ├── ambient.wav
│   ├── music.mp3
│   └── voice.wav (optional)
└── MINI_SORA_PIPELINE.md       # this documentation
```
