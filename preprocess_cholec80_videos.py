import json
from bidict import bidict
import torch
from utils.paths import path_data_cholec_full_videos_preprocessed
import os
from PIL import Image
import torchvision.transforms as T
import imageio
import tqdm
from data.cholec_scene_graph import CholecSceneGraphDataset

full_videos = "/local/scratch/Cholec80/cholec80_full_set/videos"

d = CholecSceneGraphDataset("val", num_frames=8, time_span=1.0)

unique_videos = sorted(list(set(video for video, _, _ in d.video_frame_tuples)))

for i, video in enumerate(unique_videos):
    video_idx = int(video.removeprefix("VID"))
    os.makedirs(os.path.join(path_data_cholec_full_videos_preprocessed, video), exist_ok=True)
    video_path = os.path.join(full_videos, f"video{video_idx:02d}.mp4")
    reader = imageio.get_reader(video_path)
    frames = [frame_idx for v, frame_idx, _ in d.video_frame_tuples if v == video]
    for last_frame_idx in tqdm.tqdm(frames, desc=f"Processing {video} ({i+1}/{len(unique_videos)})"):
        for frame_idx in d.delta_t + last_frame_idx:
            out_path = os.path.join(path_data_cholec_full_videos_preprocessed, video, f"{frame_idx:08d}.png")
            if os.path.exists(out_path):
                continue
            try:
                frame = reader.get_data(frame_idx)
            except Exception as e:
                print(f"Error reading frame {frame_idx} from {video}: {e}")
                continue
            frame = Image.fromarray(frame).resize((256,256))
            frame.save(out_path)
    reader.close()