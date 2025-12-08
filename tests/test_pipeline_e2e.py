"""
End-to-End Test for Mini-Sora Pipeline (with gTTS voice-over)
Run with: pytest tests/test_pipeline_e2e.py
"""

import os
from mini_sora import (
    generate_image,
    generate_video,
    refine_video,
    generate_voiceover,
    add_audio_to_video
)

def test_end_to_end_with_voice(tmp_path):
    """
    Full pipeline test:
      - text→image
      - image→video
      - refinement
      - gTTS voice-over
      - audio/video mix
    """
    # === 1️⃣ Image Generation ===
    img = generate_image(
        "A calm lake surrounded by soft morning light.",
        output_path=str(tmp_path / "frame0.png")
    )
    assert os.path.exists(img), "❌ Image not generated."

    # === 2️⃣ Video Generation ===
    vid = generate_video(
        img,
        "Camera slowly pans as the woman splashes water on her face.",
        output_video=str(tmp_path / "raw.mp4")
    )
    assert os.path.exists(vid), "❌ Video not generated."

    # === 3️⃣ Video Refinement ===
    refined = refine_video(
        vid,
        output_video=str(tmp_path / "refined.mp4")
    )
    assert os.path.exists(refined), "❌ Refinement failed."

    # === 4️⃣ Voice-Over Generation ===
    narration = (
        "A calm morning by the lake. "
        "She takes a deep breath, splashes the cool water on her face, "
        "and smiles at the rising sun."
    )
    voice_path = tmp_path / "voice.wav"
    voice = generate_voiceover(
        narration,
        lang="en",
        output_file=str(voice_path)
    )
    assert os.path.exists(voice), "❌ Voice-over not generated."

    # === 5️⃣ Audio Integration ===
    final_video = tmp_path / "final_with_voice.mp4"
    output = add_audio_to_video(
        str(refined),
        str(voice),
        str(final_video)
    )
    assert os.path.exists(output), "❌ Final video with audio not created."

    print(f"✅ E2E voice-over test completed.\nFinal output: {output}")
