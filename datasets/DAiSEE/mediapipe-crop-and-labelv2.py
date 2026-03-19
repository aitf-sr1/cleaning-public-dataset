#!/usr/bin/env python
# coding: utf-8

# # Persiapan Dataset DAiSEE: Pemrosesan Multi-Label untuk Swin Transformer
#
# Buku catatan (*notebook*) ini dirancang untuk memproses video dari dataset DAiSEE menjadi sekumpulan data gambar yang terstruktur. Dataset DAiSEE memiliki sifat **Multi-Label**, yang berarti satu wajah dapat memiliki empat nilai metrik emosi sekaligus (Boredom, Engagement, Confusion, dan Frustration).
#
# Adapun alur kerja yang akan kita lakukan meliputi:
# 1. **Ekstraksi Bingkai Video (*Frame Extraction*):** Mengambil bingkai paling awal dan paling akhir dari setiap video asli.
# 2. **Pemotongan Wajah (*Face Cropping*):** Memotong area wajah dari bingkai tersebut dengan bantuan kecerdasan buatan.
# 3. **Penyusunan Data Terpusat:** Menyimpan gambar wajah ke dalam folder kelompoknya (`Train`, `Test`, `Validation`) dan membuat file CSV baru yang sudah bersih.
#
# ### Tahap 1: Persiapan Pustaka (*Library*) dan Model Kecerdasan Buatan
# Pada tahap awal ini, kita perlu memanggil berbagai alat bantu (pustaka) yang akan digunakan di sepanjang proses. Berikut adalah penjelasan untuk masing-masing pustaka:
#
# * **`cv2` (OpenCV):** Pustaka utama untuk pemrosesan gambar komputer (*computer vision*). Kita menggunakannya untuk membuka file video `.avi`, melompat ke detik tertentu di dalam video, membaca gambar, serta memotong (menggunting) piksel area wajah secara presisi.
# * **`pandas` (`pd`):** Pustaka yang sangat kuat untuk analisis data tabular. Kita membutuhkannya untuk membaca file `Labels.csv` asli dan untuk menyusun file CSV baru yang memuat rangkuman data bersih di akhir proses.
# * **`os`:** Pustaka bawaan Python yang bertugas menjembatani program dengan sistem operasi komputer. Fungsinya adalah untuk membuat folder baru secara otomatis (`os.makedirs`) dan merangkai jalur (*path*) alamat file agar formatnya sesuai dengan sistem.
# * **`glob`:** Pustaka untuk mencari file di dalam komputer berdasarkan pola teks tertentu. Kita menggunakannya untuk melacak keberadaan file `.avi` atau `.jpg` secara otomatis tanpa harus menuliskan namanya satu per satu.
# * **`mediapipe` (`mp`):** Pustaka dari Google yang menyediakan solusi kecerdasan buatan siap pakai. Secara spesifik, kita memanggil modul `vision.FaceDetector` untuk mengenali letak wajah manusia di dalam gambar dengan cepat dan ringan.

import cv2
import glob
import os

import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any

# Tentukan alamat (path) menuju file model MediaPipe
# Model ini berfungsi sebagai "otak" untuk mendeteksi wajah
MODEL_PATH = "model/Blaze Face/blaze_face_short_range.tflite"


# ### Tahap 2: Pengaturan Lokasi Folder dan Data Label
# Sebelum proses dimulai, kita perlu memetakan lokasi penyimpanan. Kita akan menentukan lokasi video asli, lokasi folder sementara untuk hasil ekstraksi bingkai, dan direktori utama tempat dataset akhir (beserta file CSV bersihnya) akan disimpan.

# 1. Folder dataset DAiSEE asli (tempat bernaungnya file .avi)
VIDEO_FOLDER_PATH = "./DAiSEE"

# 2. Folder sementara untuk menampung gambar hasil ekstraksi video
FRAME_OUTPUT_FOLDER = "./output/Frame_output"

# 3. Folder tujuan akhir (Gambar wajah dan file CSV bersih akan disimpan di sini)
MAIN_OUTPUT_FOLDER = "./output/Cleaning_daisee_normal"

# 4. Daftar file CSV yang berisi label asli bawaan DAiSEE
CSV_PATHS = [
    "./DAiSEE/Labels/TestLabels.csv",
    "./DAiSEE/Labels/TrainLabels.csv",
    "./DAiSEE/Labels/ValidationLabels.csv",
]


# ### Tahap 3: Ekstraksi Bingkai Video dan Pembersihan Data (*Data Cleaning*)
# Pada tahap ini, program akan menelusuri file CSV dan mencari video `.avi` yang memiliki ID yang sesuai. Alih-alih memproses seluruh durasi, program akan memerintahkan OpenCV untuk melompat (*fast-forward*) ke bingkai pertama dan bingkai sebelum terakhir, lalu memotretnya menjadi gambar `.jpg`.
#
# **Logika Pembersihan Data:** Apabila terdapat ID di dalam file CSV namun video fisiknya tidak ditemukan di dalam komputer, program akan mencatatnya dan langsung melanjutkan ke ID berikutnya tanpa menimbulkan kesalahan (*error*).


def create_face_detector(model_path: str) -> Any:
    # Inisialisasi model MediaPipe
    # min_detection_confidence=0.5 memastikan bahwa model harus memiliki tingkat keyakinan minimal 50% bahwa objek tersebut adalah wajah manusia
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

    # Menghapus spasi yang tidak terlihat pada nama kolom CSV
    df.columns = df.columns.str.strip()

    print(f"{'=' * 50}\nMENGESKTRAK VIDEO: {split_name.upper()}\n{'=' * 50}")

    video_tidak_ditemukan = 0
    frame_berhasil = 0

    for _, row in df.iterrows():
        base_id = str(row["ClipID"]).replace(".avi", "")

        # Mencari keberadaan file video
        search_video = glob.glob(
            os.path.join(video_folder_path, "**", f"{base_id}.avi"), recursive=True
        )

        # Jika video tidak ditemukan, lewati dan catat
        if not search_video:
            video_tidak_ditemukan += 1
            continue

        video_path = search_video[0]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            target_frames = [0, total_frames - 2]

            output_dir = os.path.join(frame_output_folder, split_name, base_id)
            os.makedirs(output_dir, exist_ok=True)

            for frame_idx in target_frames:
                # Melompat ke bingkai yang dituju lalu memotretnya
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


# ### Tahap 4: Pemotongan Wajah dan Pembuatan CSV Multi-Label
# Ini adalah tahap akhir yang krusial. Gambar bingkai yang telah diekstrak akan diproses oleh MediaPipe. MediaPipe akan mencari koordinat wajah dan OpenCV akan memotongnya dengan memberikan sedikit ruang ekstra (*padding*) di area dahi, dagu, dan samping agar wajah tidak terpotong terlalu kaku.
#
# Wajah tersebut akan disimpan terpusat di dalam folder kelompoknya masing-masing (contoh: folder `Train`). Bersamaan dengan itu, program mendata nama file gambar yang berhasil dipotong beserta keempat nilai emosinya. Data ini dirangkum menjadi satu file CSV baru (misalnya `Cleaned_TrainLabels.csv`) yang sangat bersih dan siap dilatihkan ke model Swin Transformer.
#
# **Logika Pembersihan Data Tambahan:** Jika bingkai gambar tidak ditemukan, atau jika MediaPipe gagal menemukan wajah manusia di dalam gambar tersebut, maka data tersebut tidak akan dimasukkan ke dalam file CSV baru. Hal ini memastikan bahwa CSV akhir kita seratus persen selaras dengan jumlah gambar wajah yang tersedia.


def crop_faces_and_export_csv(
    csv_path: str,
    frame_output_folder: str,
    main_output_folder: str,
    detector: Any,
) -> None:
    split_name = get_split_name(csv_path)

    # Membuat folder penampungan gambar utama (contoh: Train, Test, atau Validation)
    split_output_folder = os.path.join(main_output_folder, split_name)
    os.makedirs(split_output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    cropped_data = []  # Wadah untuk menyusun data CSV baru
    skip_count = 0  # Menghitung gambar yang gagal dideteksi wajahnya
    missing_media_count = 0  # Menghitung label yang gambarnya tidak ada

    print(f"{'=' * 50}\nMEMOTONG WAJAH & MEMBUAT CSV: {split_name.upper()}\n{'=' * 50}")

    for _, row in df.iterrows():
        base_id = str(row["ClipID"]).replace(".avi", "")

        # Mencari gambar hasil ekstraksi dari Tahap 3
        search_pattern = os.path.join(
            frame_output_folder, split_name, base_id, f"{base_id}*.jpg"
        )
        image_files = glob.glob(search_pattern)

        # Jika gambar tidak ditemukan, lompati proses ini
        if not image_files:
            missing_media_count += 1
            continue

        for img_path in image_files:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            h, w, _ = img_bgr.shape

            # OpenCV menggunakan format BGR, sedangkan MediaPipe membutuhkan format RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = detector.detect(mp_image)

            # Jika wajah tidak terdeteksi oleh MediaPipe, lompati gambar ini
            if not detection_result.detections:
                skip_count += 1
                continue

            # Mengambil koordinat batas area wajah (Bounding Box)
            bbox = detection_result.detections[0].bounding_box
            x_min, y_min = bbox.origin_x, bbox.origin_y
            box_w, box_h = bbox.width, bbox.height

            # Menambahkan ruang ekstra (Padding) di sekitar wajah
            pad_x = int(box_w * 0.1)
            pad_top = int(box_h * 0.15)
            pad_bottom = int(box_h * 0.1)

            # Menentukan titik koordinat pemotongan final
            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_top)
            x2 = min(w, x_min + box_w + pad_x)
            y2 = min(h, y_min + box_h + pad_bottom)

            # Memotong area wajah
            cropped_face = img_bgr[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            # Menyimpan potongan wajah ke folder utama yang telah ditentukan
            img_name = os.path.basename(img_path)
            final_img_name = f"face_{img_name}"
            cropped_filepath = os.path.join(split_output_folder, final_img_name)
            cv2.imwrite(cropped_filepath, cropped_face)

            # Merekam data Multi-Label ke dalam wadah CSV baru
            cropped_data.append(
                {
                    "Image_Name": final_img_name,
                    "Boredom": int(row["Boredom"]),
                    "Engagement": int(row["Engagement"]),
                    "Confusion": int(row["Confusion"]),
                    "Frustration": int(row["Frustration"]),
                }
            )

    # Mengubah wadah data menjadi format tabel dan menyimpannya sebagai file CSV baru
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
