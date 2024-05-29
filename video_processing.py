import cv2
from scenedetect import open_video, SceneManager, VideoManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip
from transformers import CLIPProcessor, CLIPModel
import torch
import yt_dlp
import os

def process_video(video_url, description):
    # Download or load the video from the URL
    video_path = download_video(video_url)

    # Segment video into scenes
    scenes = find_scenes(video_path)

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
        if os.path.exists(final_clip_path):
            os.remove(final_clip_path)
        final_clip.write_videofile(final_clip_path)
    except Exception as e:
        return str(e)

    return final_clip_path

def find_scenes(video_path):
    # Create a video manager object for the video
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # Add ContentDetector algorithm with a threshold. Adjust threshold as needed.
    scene_manager.add_detector(ContentDetector(threshold=30))

    # Start the video manager and perform scene detection
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Obtain list of detected scenes as timecodes
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    # Collect the start and end times for each scene
    scenes = [(start.get_timecode(), end.get_timecode()) for start, end in scene_list]
    return scenes

def convert_timestamp_to_seconds(timestamp):
    """Convert a timestamp in HH:MM:SS format to seconds."""
    h, m, s = map(float, timestamp.split(':'))
    return int(h) * 3600 + int(m) * 60 + s

def analyze_scenes(video_path, scenes, description):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    best_scene = None
    highest_prob = 0.0

    for scene_id, (start_time, end_time) in enumerate(scenes):
        # Extract every 5th frame from the scene
        frames = extract_frames(video_path, start_time, end_time)

        # Analyze frames with CLIP
        for frame in frames:
            inputs = processor(text=description, images=frame, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            max_prob = max(probs[0]).item()
            if max_prob > highest_prob:
                highest_prob = max_prob
                best_scene = (start_time, end_time)

    return best_scene

def extract_frames(video_path, start_time, end_time):
    frames = []
    start_seconds = convert_timestamp_to_seconds(start_time)
    end_seconds = convert_timestamp_to_seconds(end_time)
    video_clip = VideoFileClip(video_path).subclip(start_seconds, end_seconds)

    for frame_time in range(0, int(video_clip.duration), 5):
        frame = video_clip.get_frame(frame_time)
        frames.append(frame)

    return frames

def extract_best_scene(video_path, scene):
    if scene is None:
        return VideoFileClip(video_path)  # Return the entire video if no scene is found

    start_time, end_time = scene
    start_seconds = convert_timestamp_to_seconds(start_time)
    end_seconds = convert_timestamp_to_seconds(end_time)
    video_clip = VideoFileClip(video_path).subclip(start_seconds, end_seconds)
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

