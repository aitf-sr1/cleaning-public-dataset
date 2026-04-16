import cv2
import glob
import os

import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any

MODEL_PATH = "model/Face Landmarker/face_landmarker.task"

VIDEO_FOLDER_PATH = "./DAiSEE"
FRAME_OUTPUT_FOLDER = "./output/Frame_output"
MAIN_OUTPUT_FOLDER = "./output/Cleaning_daisee_landmarks"

CSV_PATHS = [
    "./DAiSEE/Labels/TestLabels.csv",
    "./DAiSEE/Labels/TrainLabels.csv",
    "./DAiSEE/Labels/ValidationLabels.csv",
]


def create_face_mesh_detector(model_path: str) -> Any:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_split_name(csv_path: str) -> str:
    filename = os.path.basename(csv_path)
    return filename.replace("Labels.csv", "")


def extract_frames_for_split(
    csv_path: str, video_folder_path: str, frame_output_folder: str
) -> None:
    split_name = get_split_name(csv_path)
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    print(f"{'=' * 50}\nMENGESKTRAK VIDEO: {split_name.upper()}\n{'=' * 50}")

    video_tidak_ditemukan = 0
    frame_berhasil = 0

    for _, row in df.iterrows():
        base_id = str(row["ClipID"]).replace(".avi", "")

        search_video = glob.glob(
            os.path.join(video_folder_path, "**", f"{base_id}.avi"), recursive=True
        )

        if not search_video:
            video_tidak_ditemukan += 1
            continue

        video_path = search_video[0]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            target_frames = [total_frames // 2]

            output_dir = os.path.join(frame_output_folder, split_name, base_id)
            os.makedirs(output_dir, exist_ok=True)

            for frame_idx in target_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    output_filename = os.path.join(
                        output_dir, f"{base_id}_frame_{frame_idx}.jpg"
                    )
                    cv2.imwrite(output_filename, frame)
                    frame_berhasil += 1

        cap.release()

    print(f"Ekstraksi selesai! {frame_berhasil} gambar berhasil diambil.")
    print(f"Video yang tidak ditemukan: {video_tidak_ditemukan} file\n")


def extract_landmarks_and_export_csv(
    csv_path: str,
    frame_output_folder: str,
    main_output_folder: str,
    detector: Any,
) -> None:
    split_name = get_split_name(csv_path)

    os.makedirs(main_output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    landmark_data = []
    skip_count = 0
    missing_media_count = 0

    print(f"{'=' * 50}\nMENGEKSTRAK LANDMARK: {split_name.upper()}\n{'=' * 50}")

    for _, row in df.iterrows():
        base_id = str(row["ClipID"]).replace(".avi", "")

        search_pattern = os.path.join(
            frame_output_folder, split_name, base_id, f"{base_id}*.jpg"
        )
        image_files = glob.glob(search_pattern)

        if not image_files:
            missing_media_count += 1
            continue

        for img_path in image_files:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            h, w, _ = img_bgr.shape

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = detector.detect(mp_image)

            if not detection_result.face_landmarks:
                skip_count += 1
                continue

            face_landmarks = detection_result.face_landmarks[0]

            landmark_dict = {
                "Image_Name": os.path.basename(img_path),
                "Boredom": int(row["Boredom"]),
                "Engagement": int(row["Engagement"]),
                "Confusion": int(row["Confusion"]),
                "Frustration": int(row["Frustration"]),
            }

            for idx, landmark in enumerate(face_landmarks):
                landmark_dict[f"landmark_{idx}_x"] = landmark.x
                landmark_dict[f"landmark_{idx}_y"] = landmark.y

            landmark_data.append(landmark_dict)

    df_landmarks = pd.DataFrame(landmark_data)
    csv_out_path = os.path.join(main_output_folder, f"Cleaned_{split_name}Labels.csv")
    df_landmarks.to_csv(csv_out_path, index=False)

    print(f"Kelompok {split_name} selesai diproses!")
    print(
        f"Berhasil mengekstrak landmark dan mencatat ke dalam CSV: {len(df_landmarks)} data"
    )
    print(f"File CSV tersimpan di: {csv_out_path}")
    print(f"Gambar tanpa wajah yang diabaikan: {skip_count} data")
    print(f"Label tanpa gambar yang diabaikan: {missing_media_count} data\n")


def main() -> None:
    detector = create_face_mesh_detector(MODEL_PATH)

    for csv_path in CSV_PATHS:
        extract_frames_for_split(
            csv_path=csv_path,
            video_folder_path=VIDEO_FOLDER_PATH,
            frame_output_folder=FRAME_OUTPUT_FOLDER,
        )

    for csv_path in CSV_PATHS:
        extract_landmarks_and_export_csv(
            csv_path=csv_path,
            frame_output_folder=FRAME_OUTPUT_FOLDER,
            main_output_folder=MAIN_OUTPUT_FOLDER,
            detector=detector,
        )


if __name__ == "__main__":
    main()
