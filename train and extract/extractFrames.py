'''
	-The given code extracts all the frames for the entire dataset and saves these frames in the folder of the video clips.
	-Kindly have ffmpeg (https://www.ffmpeg.org/) (all credits) in order to successfully execute this script.
	-The script must in the a same directory as the Dataset Folder.
'''

import os
import subprocess

# determine base directory (where script resides) and dataset folder
base_dir = os.path.dirname(os.path.abspath(__file__))
# adjust this name if your dataset folder has a different name (Data or DataSet)
dataset_dir = os.path.join(base_dir, 'DataSet')

# make sure the dataset directory actually exists before listing
if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

dataset = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# base directory where frames will be written; change if desired
destination_base = os.path.join(base_dir, 'Frames')
# ensure base exists
os.makedirs(destination_base, exist_ok=True)

def split_video(input_path, image_name_prefix, destination_path):
    ffmpeg_path = r'D:\ffmpeg\bin\ffmpeg.exe'
    # input_path already contains full path to video file
    cmd = f'"{ffmpeg_path}" -i "{input_path}" -vf "fps=1" {image_name_prefix}%d.jpg -hide_banner'
    return subprocess.check_output(cmd, shell=True, cwd=destination_path)


for ttv in dataset:
    users_path = os.path.join(dataset_dir, ttv)
    users = os.listdir(users_path)
    for user in users:
        curr_user_path = os.path.join(users_path, user)
        currUser = os.listdir(curr_user_path)
        for extract in currUser:
            clip_dir = os.path.join(curr_user_path, extract)
            if not os.path.isdir(clip_dir):
                continue
            files = os.listdir(clip_dir)
            if not files:
                print(f"Skipping empty folder: {clip_dir}")
                continue
            clip = files[0]
            print(clip[:-4])
            # compute corresponding output folder under destination_base
            relative = os.path.relpath(clip_dir, dataset_dir)
            out_dir = os.path.join(destination_base, relative)
            os.makedirs(out_dir, exist_ok=True)
            path = out_dir + os.sep
            video_path = os.path.join(clip_dir, clip)
            split_video(video_path, clip[:-4], path)

print ("================================================================================\n")
print ("Frame Extraction Successful")