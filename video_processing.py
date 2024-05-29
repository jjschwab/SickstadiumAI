import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip
from transformers import CLIPProcessor, CLIPModel
import torch
import yt_dlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]',
        'outtmpl': 'downloaded_video.%(ext)s',
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

def find_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return scene_list

def extract_frames(video_path, scene_list):
    scene_frames = {}
    cap = cv2.VideoCapture(video_path)
    for i, (start_time, end_time) in enumerate(scene_list):
        frames = []
        first_frame = None
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if ret:
                if first_frame is None:
                    first_frame = frame
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
                    frames.append(frame)
        scene_frames[i] = (start_time, end_time, frames, first_frame)
    cap.release()
    return scene_frames

def convert_timestamp_to_seconds(timestamp):
    h, m, s = map(float, timestamp.split(':'))
    return int(h) * 3600 + int(m) * 60 + s

def classify_and_categorize_scenes(scene_frames, description_phrases):
    scene_categories = {}
    description_texts = description_phrases

    action_indices = [0]
    context_indices = list(set(range(len(description_texts))) - set(action_indices))

    for scene_id, (start_time, end_time, frames, first_frame) in scene_frames.items():
        scene_scores = [0] * len(description_texts)
        valid_frames = 0

        for frame in frames:
            image = Image.fromarray(frame[..., ::-1])
            image_input = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                text_inputs = processor(text=description_texts, return_tensors="pt", padding=True).to(device)
                text_features = model.get_text_features(**text_inputs)
                image_features = model.get_image_features(**image_input)
                logits = (image_features @ text_features.T).squeeze()
                probs = logits.softmax(dim=0)
                scene_scores = [sum(x) for x in zip(scene_scores, probs.tolist())]
                valid_frames += 1

        if valid_frames > 0:
            scene_scores = [score / valid_frames for score in scene_scores]
            action_confidence = sum(scene_scores[i] for i in action_indices) / len(action_indices)
            context_confidence = sum(scene_scores[i] for i in context_indices) / len(context_indices)

            best_description_index = scene_scores.index(max(scene_scores))
            best_description = description_texts[best_description_index]

            if action_confidence > context_confidence:
                category = "Action Scene"
                confidence = action_confidence
            else:
                category = "Context Scene"
                confidence = context_confidence

            duration = end_time.get_seconds() - start_time.get_seconds()
            scene_categories[scene_id] = {
                "category": category,
                "confidence": confidence,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "duration": duration,
                "first_frame": first_frame,
                "best_description": best_description
            }

    return scene_categories

def save_clip(video_path, scene_info, output_directory, scene_id):
    output_filename = f"scene_{scene_id+1}_{scene_info['category'].replace(' ', '_')}.mp4"
    output_filepath = os.path.join(output_directory, output_filename)

    start_seconds = convert_timestamp_to_seconds(scene_info['start_time'])
    end_seconds = convert_timestamp_to_seconds(scene_info['end_time'])

    video_clip = VideoFileClip(video_path).subclip(start_seconds, end_seconds)

    video_clip.write_videofile(output_filepath, codec='libx264', audio_codec='aac')
    video_clip.close()

    return output_filepath, scene_info['first_frame']

def process_video(video_url, description):
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    video_path = download_video(video_url)
    scenes = find_scenes(video_path)
    scene_frames = extract_frames(video_path, scenes)
    description_phrases = [description]  # Modify if multiple descriptions are needed
    scene_categories = classify_and_categorize_scenes(scene_frames, description_phrases)

    best_scene = max(scene_categories.items(), key=lambda x: x[1]['confidence'])[1]
    clip_path, first_frame = save_clip(video_path, best_scene, output_directory, 0)

    return clip_path

