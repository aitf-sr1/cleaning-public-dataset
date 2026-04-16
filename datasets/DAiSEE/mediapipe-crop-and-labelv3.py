#!/usr/bin/env python
# coding: utf-8

import cv2
import glob
import os

import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any

MODEL_PATH = "model/Blaze Face/blaze_face_short_range.tflite"

VIDEO_FOLDER_PATH = "./DAiSEE"
FRAME_OUTPUT_FOLDER = "./output/Frame_output"
MAIN_OUTPUT_FOLDER = "./output/Cleaning_daisee_normal"

CSV_PATHS = [
    "./DAiSEE/Labels/TestLabels.csv",
    "./DAiSEE/Labels/TrainLabels.csv",
    "./DAiSEE/Labels/ValidationLabels.csv",
]


def create_face_detector(model_path: str) -> Any:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=0.5
    )
    return vision.FaceDetector.create_from_options(options)


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


def crop_faces_and_export_csv(
    csv_path: str,
    frame_output_folder: str,
    main_output_folder: str,
    detector: Any,
) -> None:
    split_name = get_split_name(csv_path)

    split_output_folder = os.path.join(main_output_folder, split_name)
    os.makedirs(split_output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    cropped_data = []
    skip_count = 0
    missing_media_count = 0

    print(f"{'=' * 50}\nMEMOTONG WAJAH & MEMBUAT CSV: {split_name.upper()}\n{'=' * 50}")

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

            if not detection_result.detections:
                skip_count += 1
                continue

            bbox = detection_result.detections[0].bounding_box
            x_min, y_min = bbox.origin_x, bbox.origin_y
            box_w, box_h = bbox.width, bbox.height

            pad_x = int(box_w * 0.1)
            pad_top = int(box_h * 0.15)
            pad_bottom = int(box_h * 0.1)

            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_top)
            x2 = min(w, x_min + box_w + pad_x)
            y2 = min(h, y_min + box_h + pad_bottom)

            cropped_face = img_bgr[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            cropped_face = cv2.resize(cropped_face, (224, 224))

            img_name = os.path.basename(img_path)
            final_img_name = f"face_{img_name}"
            cropped_filepath = os.path.join(split_output_folder, final_img_name)
            cv2.imwrite(cropped_filepath, cropped_face)

            cropped_data.append(
                {
                    "Image_Name": final_img_name,
                    "Boredom": int(row["Boredom"]),
                    "Engagement": int(row["Engagement"]),
                    "Confusion": int(row["Confusion"]),
                    "Frustration": int(row["Frustration"]),
                }
            )

    df_cropped = pd.DataFrame(cropped_data)
    csv_out_path = os.path.join(main_output_folder, f"Cleaned_{split_name}Labels.csv")
    df_cropped.to_csv(csv_out_path, index=False)

    print(f"Kelompok {split_name} selesai diproses!")
    print(f"Berhasil memotong dan mencatat ke dalam CSV bersih: {len(df_cropped)} data")
    print(f"File CSV tersimpan di: {csv_out_path}")
    print(f"Gambar tanpa wajah yang diabaikan: {skip_count} data")
    print(f"Label tanpa gambar yang diabaikan: {missing_media_count} data\n")


def main() -> None:
    detector = create_face_detector(MODEL_PATH)

    for csv_path in CSV_PATHS:
        extract_frames_for_split(
            csv_path=csv_path,
            video_folder_path=VIDEO_FOLDER_PATH,
            frame_output_folder=FRAME_OUTPUT_FOLDER,
        )

    for csv_path in CSV_PATHS:
        crop_faces_and_export_csv(
            csv_path=csv_path,
            frame_output_folder=FRAME_OUTPUT_FOLDER,
            main_output_folder=MAIN_OUTPUT_FOLDER,
            detector=detector,
        )


if __name__ == "__main__":
    main()
