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
2. **Image → Video** (Waver I2V)  
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
python3 -m venv waver_env
source waver_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate pillow imageio[ffmpeg] gTTS
brew install ffmpeg

# RIFE
git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife && pip install -r requirements.txt

# FILM
pip install film
```

**Pipenv**
```bash
pip install pipenv
pipenv --python 3.12
pipenv shell

pipenv install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pipenv install diffusers transformers accelerate pillow imageio[ffmpeg] gTTS
brew install ffmpeg

# RIFE
git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife && pip install -r requirements.txt

# FILM
pipenv install film
```

Optional (for interpretation)
# RIFE
git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife && pip install -r requirements.txt

# FILM
pip install film

## Usage
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

