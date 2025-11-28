"""
Face Detector and Aligner
==========================

Module untuk detect faces dari images dan align faces ke standard size.
Menggunakan MTCNN (Multi-task Cascaded Convolutional Networks).
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger


class FaceDetector:
    """
    Face detector dan aligner menggunakan MTCNN.
    
    Attributes:
        detector: MTCNN detector instance
        target_size: Target face size setelah alignment
        min_confidence: Minimum confidence threshold
        
    Example:
        >>> detector = FaceDetector(target_size=(160, 160))
        >>> face, landmarks = detector.detect_and_align("photo.jpg")
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (160, 160),
        min_confidence: float = 0.95,
        min_face_size: int = 20
    ):
        """
        Initialize face detector.
        
        Args:
            target_size: Target size untuk aligned face (width, height)
            min_confidence: Minimum confidence untuk face detection
            min_face_size: Minimum face size dalam pixels
        """
        self.target_size = target_size
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        
        # Initialize MTCNN detector
        # Note: MTCNN from mtcnn library has different parameters than facenet_pytorch
        self.detector = MTCNN()
        
        self.logger = setup_logger("FaceDetector")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces dalam image.
        
        Args:
            image: Image array (RGB format)
            
        Returns:
            List of detection dictionaries dengan keys:
            - 'box': [x, y, width, height]
            - 'confidence': confidence score
            - 'keypoints': facial landmarks
        """
        detections = self.detector.detect_faces(image)
        
        # Filter by confidence
        filtered = [
            det for det in detections 
            if det['confidence'] >= self.min_confidence
        ]
        
        return filtered
    
    def align_face(
        self, 
        image: np.ndarray, 
        box: List[int],
        margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Crop dan resize face berdasarkan bounding box.
        
        Args:
            image: Image array
            box: Bounding box [x, y, width, height]
            margin: Margin around face (proporsi)
            
        Returns:
            Aligned face array atau None jika gagal
        """
        x, y, width, height = box
        
        # Add margin
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + width + margin_x)
        y2 = min(image.shape[0], y + height + margin_y)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(
            face, 
            self.target_size, 
            interpolation=cv2.INTER_AREA
        )
        
        return face_resized
    
    def process_image(
        self, 
        image_path: str,
        save_path: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Process single image: detect dan align face.
        
        Args:
            image_path: Path ke input image
            save_path: Path untuk save aligned face (optional)
            
        Returns:
            Tuple of (aligned_face, num_faces_detected)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None, 0
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detect_faces(image_rgb)
            
            if len(detections) == 0:
                self.logger.warning(f"No face detected in: {Path(image_path).name}")
                return None, 0
            
            if len(detections) > 1:
                self.logger.warning(
                    f"Multiple faces ({len(detections)}) detected in: {Path(image_path).name}. "
                    f"Using face with highest confidence."
                )
            
            # Use face with highest confidence
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # Align face
            aligned_face = self.align_face(image_rgb, best_detection['box'])
            
            if aligned_face is None:
                self.logger.error(f"Failed to align face in: {Path(image_path).name}")
                return None, len(detections)
            
            # Save if path provided
            if save_path:
                # Convert RGB to BGR for saving
                face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, face_bgr)
            
            return aligned_face, len(detections)
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None, 0


def detect_and_align_faces(
    source_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (160, 160),
    min_confidence: float = 0.95
) -> Tuple[int, int, int]:
    """
    Detect dan align faces untuk seluruh dataset.
    
    Args:
        source_dir: Source directory dengan subdirectories per student
        output_dir: Output directory untuk aligned faces
        target_size: Target size untuk aligned faces
        min_confidence: Minimum confidence untuk detection
        
    Returns:
        Tuple of (total_images, successful, failed)
        
    Example:
        >>> total, success, failed = detect_and_align_faces(
        ...     "dataset/Train", 
        ...     "dataset/Train_Aligned"
        ... )
    """
    logger = setup_logger("FaceAlignment")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return 0, 0, 0
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = FaceDetector(
        target_size=target_size,
        min_confidence=min_confidence
    )
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    logger.info(f"Starting face detection and alignment...")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target size: {target_size}")
    
    total_images = 0
    successful = 0
    failed = 0
    
    # Process each student directory
    student_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    for student_dir in student_dirs:
        student_name = student_dir.name
        logger.info(f"\nProcessing: {student_name}")
        
        # Create output directory for this student
        student_output_dir = output_path / student_name
        student_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images for this student
        image_files = [
            f for f in student_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        for image_file in image_files:
            total_images += 1
            
            # Create output path
            output_file = student_output_dir / f"{image_file.stem}_aligned{image_file.suffix}"
            
            # Process image
            aligned_face, num_faces = detector.process_image(
                str(image_file),
                str(output_file)
            )
            
            if aligned_face is not None:
                successful += 1
                logger.info(f"  ✓ {image_file.name}")
            else:
                failed += 1
                logger.warning(f"  ✗ {image_file.name}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Face Detection & Alignment Summary:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Successfully processed: {successful} ({successful/total_images*100:.1f}%)")
    logger.info(f"  Failed: {failed} ({failed/total_images*100:.1f}%)")
    logger.info("=" * 60)
    
    return total_images, successful, failed


if __name__ == "__main__":
    """
    Run face detection and alignment pada dataset.
    
    Usage:
        python -m src.preprocessing.face_detector
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and align faces in dataset")
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/Train",
        help="Source directory with student subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/Train_Aligned",
        help="Output directory for aligned faces"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=160,
        help="Target face size (square)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Minimum confidence threshold"
    )
    
    args = parser.parse_args()
    
    # Run face detection and alignment
    total, success, failed = detect_and_align_faces(
        args.source,
        args.output,
        target_size=(args.size, args.size),
        min_confidence=args.confidence
    )
    
    if success == total:
        print("\n✓ All faces detected and aligned successfully!")
    elif success > 0:
        print(f"\n⚠ Processed {success}/{total} images successfully")
        print(f"  {failed} images failed (no face detected or processing error)")
    else:
        print("\n✗ No faces were successfully processed")
