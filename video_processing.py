import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip
from transformers import CLIPProcessor, CLIPModel
import torch
import yt_dlp
from PIL import Image
import uuid
from torchvision import models, transforms
from torch.nn import functional as F
from cachetools import cached, TTLCache
import numpy as np


categories = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]

#initializing CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#initializing ZG placeholder
resnet50 = models.resnet50(pretrained=True).eval().to(device)

#initialize caches
scene_cache = TTLCache(maxsize=100, ttl=86400)  # cache up to 100 items, each for 1 day
frame_cache = TTLCache(maxsize=100, ttl=86400)
analysis_cache = TTLCache(maxsize=100, ttl=86400)


def cache_info_decorator(func, cache):
    """Decorator to add caching and logging to a function."""
    key_func = lambda *args, **kwargs: "_".join(map(str, args))  # Simple key func based on str(args)

    @cached(cache, key=key_func)
    def wrapper(*args, **kwargs):
        key = key_func(*args, **kwargs)
        if key in cache:
            logging.info(f"Cache hit for key: {key}")
        else:
            logging.info(f"Cache miss for key: {key}. Caching result.")
        return func(*args, **kwargs)
    return wrapper

    
def classify_frame(frame):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.fromarray(frame))
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Use the globally loaded ResNet-50 model
    with torch.no_grad():
        output = resnet50(input_batch)
        probabilities = F.softmax(output[0], dim=0)

    results_array = np.array([probabilities[i].item() for i in range(len(categories))])
    return results_array


def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]',
        'outtmpl': f'temp_videos/{uuid.uuid4()}_video.%(ext)s',
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        video_filename = ydl.prepare_filename(result)
        safe_filename = sanitize_filename(video_filename)
        if os.path.exists(video_filename) and video_filename != safe_filename:
            os.rename(video_filename, safe_filename)
        return safe_filename

def sanitize_filename(filename):
    return "".join([c if c.isalnum() or c in " .-_()" else "_" for c in filename])

@cache_info_decorator
def find_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=33))  # Adjusted threshold for finer segmentation
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    scenes = [(start.get_timecode(), end.get_timecode()) for start, end in scene_list]
    return scenes

def convert_timestamp_to_seconds(timestamp):
    h, m, s = map(float, timestamp.split(':'))
    return int(h) * 3600 + int(m) * 60 + s

@cache_info_decorator
def extract_frames(video_path, start_time, end_time):
    frames = []
    start_seconds = convert_timestamp_to_seconds(start_time)
    end_seconds = convert_timestamp_to_seconds(end_time)
    video_clip = VideoFileClip(video_path).subclip(start_seconds, end_seconds)
    # Extract more frames: every frame in the scene
    for frame_time in range(0, int(video_clip.duration * video_clip.fps), int(video_clip.fps / 10)):
        frame = video_clip.get_frame(frame_time / video_clip.fps)
        frames.append(frame)
    return frames

@cache_info_decorator
def analyze_scenes(video_path, scenes, description):
    scene_scores = []
    negative_descriptions = [
        "black screen",
        "Intro text for a video",
        "dark scene without much contrast",
        "No people are in this scene",
        "A still shot of natural scenery",
        "Still-camera shot of a person's face"
    ]

    text_inputs = processor(text=[description] + negative_descriptions, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs).detach()
    positive_feature, negative_features = text_features[0], text_features[1:]

    for scene_num, (start_time, end_time) in enumerate(scenes):
        frames = extract_frames(video_path, start_time, end_time)
        if not frames:
            print(f"Scene {scene_num + 1}: Start={start_time}, End={end_time} - No frames extracted")
            continue

        scene_prob = 0.0
        sentiment_distributions = np.zeros(8)  # Assuming there are 8 sentiments
        for frame in frames:
            image = Image.fromarray(frame[..., ::-1])
            image_input = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**image_input).detach()
                positive_similarity = torch.cosine_similarity(image_features, positive_feature.unsqueeze(0)).squeeze().item()
                negative_similarities = torch.cosine_similarity(image_features, negative_features).squeeze().mean().item()
                scene_prob += positive_similarity - negative_similarities
            
            frame_sentiments = classify_frame(frame)
            sentiment_distributions += np.array(frame_sentiments)

        sentiment_distributions /= len(frames)  # Normalize to get average probabilities
        sentiment_percentages = {category: round(prob * 100, 2) for category, prob in zip(categories, sentiment_distributions)}
        scene_prob /= len(frames)
        scene_duration = convert_timestamp_to_seconds(end_time) - convert_timestamp_to_seconds(start_time)
        print(f"Scene {scene_num + 1}: Start={start_time}, End={end_time}, Probability={scene_prob}, Duration={scene_duration}, Sentiments: {sentiment_percentages}")

        scene_scores.append((scene_prob, start_time, end_time, scene_duration, sentiment_percentages))

        # Sort scenes by confidence, highest first
        scene_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Select the longest scene from the top 3 highest confidence scenes
        top_3_scenes = scene_scores[:3]  # Get the top 3 scenes
        best_scene = max(top_3_scenes, key=lambda x: x[3])  # Find the longest scene from these top 3


    if best_scene:
        print(f"Best Scene: Start={best_scene[1]}, End={best_scene[2]}, Probability={best_scene[0]}, Duration={best_scene[3]}, Sentiments: {best_scene[4]}")
        return (best_scene[1], best_scene[2]), best_scene[4]  # Returning a tuple with scene times and sentiments
    else:
        print("No suitable scene found")
        return None, {}


def extract_best_scene(video_path, scene):
    if scene is None:
        return None

    start_time, end_time = scene
    start_seconds = convert_timestamp_to_seconds(start_time)
    end_seconds = convert_timestamp_to_seconds(end_time)
    video_clip = VideoFileClip(video_path).subclip(start_seconds, end_seconds)
    return video_clip

def process_video(video_url, description):
    video_path = download_video(video_url)
    scenes = find_scenes(video_path)
    best_scene = analyze_scenes(video_path, scenes, description)
    final_clip = extract_best_scene(video_path, best_scene)

    if final_clip:
        # Assuming final_clip is a MoviePy VideoFileClip object
        frame = np.array(final_clip.get_frame(0))  # Get the first frame at t=0 seconds
        frame_classification = classify_frame(frame)  # Classify the frame
        print("Frame classification probabilities:", frame_classification)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        final_clip_path = os.path.join(output_dir, f"{uuid.uuid4()}_final_clip.mp4")
        final_clip.write_videofile(final_clip_path, codec='libx264', audio_codec='aac')
        cleanup_temp_files()
        return final_clip_path

    return None


def cleanup_temp_files():
    temp_dir = 'temp_videos'
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")