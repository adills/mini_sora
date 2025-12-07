"""
End-to-End Test for Mini-Sora Pipeline
Run with: pytest tests/test_pipeline_e2e.py
"""

import os
from mini_sora import generate_image, generate_video, refine_video

def test_end_to_end(tmp_path):
    img = generate_image("A small waterfall in a forest", output_path=str(tmp_path / "frame0.png"))
    assert os.path.exists(img)

    vid = generate_video(img, "Camera slowly pans upward revealing sunlight", output_video=str(tmp_path / "raw.mp4"))
    assert os.path.exists(vid)

    final = refine_video(vid, output_video=str(tmp_path / "final.mp4"))
    assert os.path.exists(final)

    print(f"✅ E2E test completed. Output at: {final}")
