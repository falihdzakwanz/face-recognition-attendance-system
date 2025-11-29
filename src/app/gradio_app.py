"""
Gradio Desktop Application
===========================

Real-time face recognition untuk attendance system dengan:
- Webcam capture
- Face detection dan recognition
- Attendance logging
- Student database management
"""

import gradio as gr
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.preprocessing.face_detector import FaceDetector
from src.evaluation.evaluate import load_trained_model


class FaceRecognitionApp:
    """
    Face Recognition Application.

    Attributes:
        model: Trained model
        detector: MTCNN face detector
        class_names: Student names
        attendance_log: Attendance records
    """

    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize application.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to config file
        """
        self.logger = setup_logger("FaceRecApp")
        self.config = load_config(config_path)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Load class names
        train_dir = Path(self.config["paths"]["train_dir"])
        self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.class_names)

        # Load model
        self.logger.info(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()

        # Initialize face detector
        target_size = tuple(self.config["preprocessing"]["face_alignment"]["target_size"])
        self.detector = FaceDetector(
            target_size=target_size,
            min_confidence=self.config["preprocessing"]["face_detection"]["confidence_threshold"]
        )

        # Attendance tracking
        self.attendance_log_path = (
            Path(self.config["paths"]["output_dir"]) / "attendance.csv"
        )
        self.attendance_log = self._load_attendance_log()
        self.attendance_cooldown = timedelta(minutes=5)  # Prevent duplicate entries

        self.logger.info("✓ Application initialized")

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load trained model."""
        from src.models.cnn_facenet import create_model
        from src.models.transformer_deit import create_transformer_model

        # Determine model type
        if "cnn" in checkpoint_path.lower() or "facenet" in checkpoint_path.lower():
            model = create_model(self.config, self.num_classes, model_type="arcface")
        else:
            model = create_transformer_model(
                self.config, self.num_classes, model_type="deit"
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(self.device)

    def _load_attendance_log(self) -> pd.DataFrame:
        """Load attendance log."""
        if self.attendance_log_path.exists():
            return pd.read_csv(self.attendance_log_path)
        else:
            return pd.DataFrame(
                columns=["timestamp", "student_name", "confidence", "status"]
            )

    def _save_attendance_log(self):
        """Save attendance log."""
        self.attendance_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.attendance_log.to_csv(self.attendance_log_path, index=False)

    def _check_duplicate(self, student_name: str) -> bool:
        """
        Check apakah student sudah absen dalam cooldown period.

        Args:
            student_name: Student name

        Returns:
            True jika duplicate, False otherwise
        """
        if len(self.attendance_log) == 0:
            return False

        # Filter records untuk student ini
        student_records = self.attendance_log[
            self.attendance_log["student_name"] == student_name
        ]

        if len(student_records) == 0:
            return False

        # Check last attendance time
        last_attendance = pd.to_datetime(student_records.iloc[-1]["timestamp"])
        now = datetime.now()

        return (now - last_attendance) < self.attendance_cooldown

    def _add_attendance_record(self, student_name: str, confidence: float):
        """Add attendance record."""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_name": student_name,
            "confidence": f"{confidence:.4f}",
            "status": "Present",
        }

        # Append to dataframe
        self.attendance_log = pd.concat(
            [self.attendance_log, pd.DataFrame([record])], ignore_index=True
        )

        self._save_attendance_log()

        self.logger.info(f"✓ Attendance recorded: {student_name} ({confidence:.2%})")

    @torch.no_grad()
    def predict_face(
        self, face_image: np.ndarray, confidence_threshold: float = 0.55
    ) -> Tuple[str, float]:
        """
        Predict student dari face image.

        Args:
            face_image: Face image (RGB)
            confidence_threshold: Confidence threshold

        Returns:
            Tuple of (student_name, confidence)
        """
        # Preprocess
        if "cnn" in str(self.model.__class__).lower():
            size = 160
        else:
            size = 224

        face_resized = cv2.resize(face_image, (size, size))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        face_tensor = (face_tensor - mean) / std

        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        # Predict
        outputs = self.model(face_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        confidence = confidence.item()
        student_name = self.class_names[pred_idx.item()]

        if confidence < confidence_threshold:
            return "Unknown", confidence

        return student_name, confidence

    def process_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.55,
        auto_mark: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """
        Process single frame.

        Args:
            frame: Input frame (BGR)
            confidence_threshold: Confidence threshold
            auto_mark: Automatically mark attendance

        Returns:
            Tuple of (annotated_frame, info_text)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detector.detect_faces(rgb_frame)

        info_lines = []

        if len(faces) == 0:
            info_lines.append("No face detected")
            return frame, "\n".join(info_lines)

        # Process each face
        for i, detection in enumerate(faces):
            # Extract bbox [x, y, width, height]
            x, y, w, h = detection['box']
            x1, y1, x2, y2 = x, y, x+w, y+h
            
            # Align face
            aligned_face = self.detector.align_face(rgb_frame, detection['box'])
            
            if aligned_face is None:
                continue

            # Predict
            student_name, conf = self.predict_face(aligned_face, confidence_threshold)

            # Draw bbox
            if student_name == "Unknown":
                color = (0, 0, 255)  # Red
                label = f"Unknown ({conf:.2%})"
            else:
                color = (0, 255, 0)  # Green
                label = f"{student_name} ({conf:.2%})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Info
            info_lines.append(f"Face {i+1}: {label}")

            # Auto mark attendance
            if auto_mark and student_name != "Unknown":
                if not self._check_duplicate(student_name):
                    self._add_attendance_record(student_name, conf)
                    info_lines.append(f"✓ Attendance marked for {student_name}")
                else:
                    info_lines.append(f"⚠ {student_name} already marked recently")

        return frame, "\n".join(info_lines)

    def get_attendance_summary(self) -> pd.DataFrame:
        """Get attendance summary."""
        if len(self.attendance_log) == 0:
            return pd.DataFrame()

        # Group by date
        self.attendance_log["date"] = pd.to_datetime(
            self.attendance_log["timestamp"]
        ).dt.date

        summary = (
            self.attendance_log.groupby(["date", "student_name"])
            .size()
            .reset_index(name="count")
        )

        return summary

    def get_student_database(self) -> pd.DataFrame:
        """Get student database."""
        return pd.DataFrame(
            {
                "No": range(1, len(self.class_names) + 1),
                "Student Name": self.class_names,
            }
        )


def create_gradio_interface(
    model_path: str, config_path: str = "config.yaml"
) -> gr.Blocks:
    """
    Create Gradio interface.

    Args:
        model_path: Path to trained model
        config_path: Path to config file

    Returns:
        Gradio Blocks interface
    """
    app = FaceRecognitionApp(model_path, config_path)

    with gr.Blocks(title="Face Recognition Attendance System") as demo:
        gr.Markdown("# 🎓 Face Recognition Attendance System")
        gr.Markdown("Real-time face recognition untuk presensi mahasiswa")

        with gr.Tab("📸 Live Recognition"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image or Use Webcam", type="numpy"
                    )

                    with gr.Row():
                        confidence_slider = gr.Slider(
                            minimum=0.3,
                            maximum=0.99,
                            value=0.3,
                            step=0.05,
                            label="Confidence Threshold",
                        )
                        auto_mark = gr.Checkbox(
                            label="Auto Mark Attendance", value=False
                        )

                    recognize_btn = gr.Button("🔍 Recognize", variant="primary")

                with gr.Column():
                    output_image = gr.Image(label="Result")
                    output_text = gr.Textbox(label="Recognition Info", lines=5)

            # Process function
            def process_image(img, conf, auto):
                if img is None:
                    return None, "No image provided"
                result_img, info = app.process_frame(img, conf, auto)
                return result_img, info

            recognize_btn.click(
                fn=process_image,
                inputs=[image_input, confidence_slider, auto_mark],
                outputs=[output_image, output_text],
            )

        with gr.Tab("📊 Attendance Log"):
            gr.Markdown("## Attendance Records")

            refresh_btn = gr.Button("🔄 Refresh Log")
            attendance_table = gr.Dataframe(
                value=app.attendance_log, label="Attendance Log"
            )

            refresh_btn.click(fn=lambda: app.attendance_log, outputs=attendance_table)

        with gr.Tab("👥 Student Database"):
            gr.Markdown("## Registered Students")

            student_table = gr.Dataframe(
                value=app.get_student_database(),
                label=f"Total Students: {app.num_classes}",
            )

            gr.Markdown(f"**Total Registered Students:** {app.num_classes}")

        with gr.Tab("📈 Statistics"):
            gr.Markdown("## Attendance Statistics")

            summary_btn = gr.Button("📊 Generate Summary")
            summary_table = gr.Dataframe(label="Summary")

            summary_btn.click(fn=app.get_attendance_summary, outputs=summary_table)

    return demo


if __name__ == "__main__":
    """Launch application."""
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition App")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create public shareable link"
    )

    args = parser.parse_args()

    # Create interface
    demo = create_gradio_interface(args.model, args.config)

    # Launch
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=args.share, show_error=True
    )


def launch_gradio_app(config: dict, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """
    Launch Gradio application dengan auto-loading latest model.
    
    Args:
        config: Configuration dictionary
        share: Create public shareable link (for HF Spaces set to False)
        server_name: Server host (default: 0.0.0.0 for all interfaces)
        server_port: Server port (default: 7860)
    """
    logger = setup_logger("AppLauncher")
    
    # Find latest trained CNN model
    models_dir = Path(config["paths"]["models_dir"])
    cnn_checkpoints = list(models_dir.glob("cnn_*/best_model.pth"))
    
    if not cnn_checkpoints:
        logger.error("No trained CNN model found!")
        logger.info("Please train a model first: python main.py --train")
        return
    
    # Use latest model
    latest_model = max(cnn_checkpoints, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using model: {latest_model.parent.name}")
    
    # Create and launch interface
    demo = create_gradio_interface(str(latest_model), "config.yaml")
    
    logger.info("=" * 60)
    logger.info("🚀 Launching Face Recognition Application")
    logger.info(f"Server: http://localhost:{server_port}")
    if share:
        logger.info("Public link will be generated...")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    demo.launch(share=share, server_port=server_port, server_name=server_name, show_error=True)
