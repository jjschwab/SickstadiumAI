import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip, concatenate_videoclips
from transformers import CLIPProcessor, CLIPModel
import torch
import yt_dlp
import os

def process_video(video_url, description):
    # Download or load the video from the URL
    video_path = download_video(video_url)

    # Segment video into scenes
    scenes = detect_scenes(video_path)

    # Extract frames and analyze with CLIP model
    best_scene = analyze_scenes(video_path, scenes, description)

    # Extract the best scene into a final clip
    final_clip = extract_best_scene(video_path, best_scene)

    # Ensure the output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    final_clip_path = os.path.join(output_dir, "final_clip.mp4")

    # Save and return the final clip
    try:
        final_clip.write_videofile(final_clip_path)
    except Exception as e:
        return str(e)

    return final_clip_path

def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    return scene_list

def analyze_scenes(video_path, scenes, description):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    best_scene = None
    highest_prob = 0.0

    for scene in scenes:
        # Extract every 5th frame from the scene
        frames = extract_frames(video_path, scene)

        # Analyze frames with CLIP
        for frame in frames:
            inputs = processor(text=description, images=frame, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            max_prob = max(probs[0]).item()
            if max_prob > highest_prob:
                highest_prob = max_prob
                best_scene = scene

    return best_scene

def extract_frames(video_path, scene):
    frames = []
    start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
    video_clip = VideoFileClip(video_path)

    for frame_num in range(start_frame, end_frame, 5):
        frame = video_clip.get_frame(frame_num / video_clip.fps)
        frames.append(frame)

    return frames

def extract_best_scene(video_path, scene):
    if scene is None:
        return VideoFileClip(video_path)  # Return the entire video if no scene is found

    start_time = scene[0].get_seconds()
    end_time = scene[1].get_seconds()
    video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
    return video_clip

def download_video(video_url):
    ydl_opts = {
        'format': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]',
        'outtmpl': 'downloaded_video.%(ext)s',
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_file = ydl.prepare_filename(info_dict)
    
    return video_file
