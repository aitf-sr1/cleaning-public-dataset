import os
import cv2
import argparse
from retinaface import RetinaFace
import numpy as np

# Global detector model - initialized once
_detector = None

def get_detector():
    """Initialize RetinaFace detector (singleton pattern)."""
    global _detector
    if _detector is None:
        print("Initializing RetinaFace...")

        # Simple GPU check (will work even without TensorFlow installed)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append(str(gpu))
                print(f"✅ TensorFlow GPU detected: {', '.join(gpu_info)}")
                print("   RetinaFace will use GPU for faster processing")
            else:
                print("⚠️  No GPU detected - using CPU (slower processing)")
        except ImportError:
            print("ℹ️  TensorFlow not available - GPU detection skipped")
            print("   RetinaFace will auto-detect GPU if available")

        # RetinaFace will auto-load model on first detect() call
        _detector = True

        print("RetinaFace ready.")

    return _detector

def detect_and_crop_faces(frame_dir, output_dir, target_size=(224, 224),
                          confidence_threshold=0.9, frame_step=1):
    """
    Detect faces in frames using RetinaFace and crop/resize them.

    Args:
        frame_dir: Directory containing extracted frames
        output_dir: Directory to save cropped face images
        target_size: Target size for resized faces (default: 224x224)
        confidence_threshold: Minimum confidence score for face detection (default: 0.9)
        frame_step: Process every nth frame to reduce dataset size (default: 1, means all frames)
    """
    if not os.path.isdir(frame_dir):
        print(f"Frame directory not found: {frame_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get RetinaFace detector (loaded once globally)
    detector = get_detector()

    # Get list of frame files
    all_frame_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Apply frame sampling if frame_step > 1
    frame_files = all_frame_files[::frame_step]

    if not frame_files:
        print(f"No frame images found in: {frame_dir}")
        return

    print(f"Found {len(all_frame_files)} total frame(s), processing {len(frame_files)} frame(s) (step={frame_step})...")

    face_count = 0
    skipped_count = 0

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)

        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to read: {frame_file}")
                continue

            # Detect faces using file path (some RetinaFace versions require path, not numpy array)
            faces = RetinaFace.detect_faces(frame_path)

            # Filter faces for DAiSEE dataset quality - find best face only
            if faces and isinstance(faces, dict):
                # Find the best face: largest area + high confidence + closest to center
                best_face = None
                best_score = 0

                frame_height, frame_width = frame.shape[:2]
                frame_center_x = frame_width // 2
                frame_center_y = frame_height // 2

                for face_id, face_info in faces.items():
                    confidence = face_info.get("score", 0)

                    # Skip low confidence faces
                    if confidence < confidence_threshold:
                        continue

                    x1, y1, x2, y2 = face_info["facial_area"]
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_area = face_width * face_height

                    # Skip very small faces (less than 3% of frame)
                    frame_area = frame_height * frame_width
                    if face_area < 0.03 * frame_area:
                        continue

                    # Calculate distance from frame center
                    face_center_x = (x1 + x2) // 2
                    face_center_y = (y1 + y2) // 2
                    distance_from_center = ((face_center_x - frame_center_x) ** 2 +
                                          (face_center_y - frame_center_y) ** 2) ** 0.5

                    # Skip faces too far from center (more than 40% of frame width)
                    if distance_from_center > frame_width * 0.4:
                        continue

                    # Score faces by combination of area and confidence
                    # Larger faces and higher confidence get better scores
                    face_score = face_area * confidence

                    if face_score > best_score:
                        best_score = face_score
                        best_face = face_info

                # Process only the best face found
                if best_face is not None:
                    face_info = best_face
                    confidence = face_info.get("score", 0)

                    # Extract face coordinates
                    facial_area = face_info["facial_area"]
                    x1, y1, x2, y2 = facial_area

                    # Add some padding to include context
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)

                    # Crop face region
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    # Resize to target size
                    face_resized = cv2.resize(face_crop, target_size)

                    # Save cropped face
                    base_name = os.path.splitext(frame_file)[0]
                    output_file = os.path.join(output_dir, f"{base_name}_face.jpg")
                    cv2.imwrite(output_file, face_resized)
                    face_count += 1

        except Exception as e:
            print(f"Error processing {frame_file}: {e}")

    print(f"\nTotal faces extracted and resized: {face_count}")
    print(f"Faces skipped (low confidence): {skipped_count}")
    print(f"Output directory: {output_dir}")

def batch_process_frames(dataset_base_dir, output_base_dir, target_size=(224, 224),
                         confidence_threshold=0.9, frame_step=1):
    """
    Batch process frames from entire dataset structure.

    Args:
        dataset_base_dir: Base directory containing extracted frames (e.g., Frames/)
        output_base_dir: Base output directory for cropped faces
        target_size: Target size for resized faces
        confidence_threshold: Minimum confidence score for face detection
        frame_step: Process every nth frame to reduce dataset size (1=all, 3=every 3rd frame, etc)
    """
    if not os.path.isdir(dataset_base_dir):
        print(f"Dataset directory not found: {dataset_base_dir}")
        return

    os.makedirs(output_base_dir, exist_ok=True)

    # Load model once for entire batch
    print(f"Starting batch processing with frame_step={frame_step}...")
    detector = get_detector()

    # GPU is ready for inference (TensorFlow will handle it)

    # Iterate through dataset structure (Test/Train/Validation)
    for split in sorted(os.listdir(dataset_base_dir)):
        split_path = os.path.join(dataset_base_dir, split)
        if not os.path.isdir(split_path):
            continue

        print(f"\nProcessing {split} split...")

        for user in sorted(os.listdir(split_path)):
            user_path = os.path.join(split_path, user)
            if not os.path.isdir(user_path):
                continue

            for clip in sorted(os.listdir(user_path)):
                clip_path = os.path.join(user_path, clip)
                if not os.path.isdir(clip_path):
                    continue

                # Create corresponding output directory
                output_path = os.path.join(output_base_dir, split, user, clip)
                print(f" Processing: {split}/{user}/{clip}")
                detect_and_crop_faces(clip_path, output_path, target_size, confidence_threshold, frame_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and crop faces from frames using RetinaFace.')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                        help='Process single frame directory or batch process entire dataset')
    parser.add_argument('--frame_dir', help='Directory containing frames (for single mode)')
    parser.add_argument('--output_dir', help='Output directory for cropped faces (for single mode)')
    parser.add_argument('--dataset_dir', help='Base dataset directory containing frames (for batch mode)')
    parser.add_argument('--output_base', help='Base output directory for batch processing')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                        help='Target size for resized faces (default: 224 224)')
    parser.add_argument('--confidence', type=float, default=0.9,
                        help='Confidence threshold for face detection (default: 0.9)')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Process every nth frame (1=all, 3=every 3rd, 5=every 5th, etc). Use 3-5 for large datasets')

    args = parser.parse_args()
    target_size = tuple(args.size)

    if args.mode == 'single':
        if not args.frame_dir or not args.output_dir:
            print("Error: --frame_dir and --output_dir required for single mode")
            exit(1)
        detect_and_crop_faces(args.frame_dir, args.output_dir, target_size,
                            confidence_threshold=args.confidence, frame_step=args.frame_step)

    else:  # batch mode
        if not args.dataset_dir or not args.output_base:
            print("Error: --dataset_dir and --output_base required for batch mode")
            exit(1)
        batch_process_frames(args.dataset_dir, args.output_base, target_size,
                           confidence_threshold=args.confidence, frame_step=args.frame_step)