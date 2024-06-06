import gradio as gr
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def save_and_display_video(video_file):
    if video_file is None:
        return None, "No video uploaded."

    try:
        if len(video_file) > 0:
            # Save the binary content to a file
            file_path = 'uploaded_video.mp4'  # Assuming .mp4 for simplicity
            with open(file_path, 'wb') as f:
                f.write(video_file)

            scenes = find_scenes(file_path)
            scene_info = ', '.join([f"Start: {start.get_seconds()}, End: {end.get_seconds()}" for start, end in scenes])
            if scenes:
                return file_path, f"Video uploaded and displayed successfully. Scenes detected: {scene_info}"
            else:
                return file_path, "Video uploaded but no scenes were detected."
        else:
            return None, "Uploaded file is empty."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def find_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))
    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode())
    video_manager.release()

    return scene_list

with gr.Blocks() as demo:
    with gr.Column():
        video_file = gr.File(label="Upload Video File", type="binary", file_types=["video"], interactive=True)
        output_video = gr.Video()
        output_message = gr.Textbox(label="Output Message", lines=4)
        submit_button = gr.Button("Display Video")
        submit_button.click(
            fn=save_and_display_video, 
            inputs=video_file, 
            outputs=[output_video, output_message]
        )

demo.launch()
