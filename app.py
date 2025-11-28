"""
Hugging Face Spaces Entry Point
================================

Face Recognition Presensi Mahasiswa
Deploy to: https://huggingface.co/spaces/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.app.gradio_app import launch_gradio_app
from src.utils.config_loader import load_config

if __name__ == "__main__":
    # Load config
    config = load_config("config.yaml")
    
    # Launch with public sharing enabled
    launch_gradio_app(
        config=config,
        share=False,  # HF Spaces handles sharing
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860  # Default Gradio port
    )
