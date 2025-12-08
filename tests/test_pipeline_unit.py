"""
Unit Tests for Mini-Sora Pipeline Functions
Run with: pytest tests/test_pipeline_unit.py
"""

import os
import pytest
from mini_sora import generate_image, generate_voiceover

@pytest.mark.unit
def test_generate_image(tmp_path):
    os.environ["MINI_SORA_TEST_MODE"] = "1"
    prompt = "A serene lake under golden sunrise"
    img_path = tmp_path / "test_img.png"
    result = generate_image(prompt, str(img_path))
    assert os.path.exists(result), "Image generation failed"

@pytest.mark.unit
def test_generate_voiceover(tmp_path):
    os.environ["MINI_SORA_TEST_MODE"] = "1"
    text = "Testing voice-over generation."
    out_path = tmp_path / "voice.wav"
    result = generate_voiceover(text, lang="en", output_file=str(out_path))
    assert os.path.exists(result), "Voice-over generation failed"
