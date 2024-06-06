import gradio as gr
from video_processing import process_video, download_video, find_scenes, analyze_scenes, extract_best_scene, cleanup_temp_files
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable
import uuid
import os

class CustomTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.orange,
        secondary_hue: colors.Color | str = colors.orange,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Sora"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Sora"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="radial-gradient(circle at center, rgba(235, 87, 38, 1) 0%, rgba(235, 87, 38, 0) 70%), radial-gradient(#eb5726 1px, transparent 1px)",
            body_text_color="#282828",
            block_background_fill="#ffffff",
            block_title_text_color="#eb5726",
            block_label_text_color="#eb5726",
            button_primary_background_fill="#eb5726",
            button_primary_text_color="#ffffff",
        )

custom_theme = CustomTheme()

def save_uploaded_file(uploaded_file):
    upload_dir = "uploaded_videos"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{uuid.uuid4()}.mp4")
    with open(file_path, "wb") as f:
        f.write(uploaded_file)
    return file_path
    
def display_results(video_url, video_file, description):
    if video_url:
        video_path = download_video(video_url)
    elif video_file:
        video_path = save_uploaded_file(video_file)
    else:
        return "No video provided", None, "No data"

    scenes = find_scenes(video_path)
    if not scenes:
        return "No scenes detected", None, "No data"

    best_scene_times, sentiments = analyze_scenes(video_path, scenes, description)
    if not best_scene_times:
        return "No matching scene found", None, "No data"

    final_clip = extract_best_scene(video_path, best_scene_times)
    if final_clip:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        final_clip_path = os.path.join(output_dir, f"{uuid.uuid4()}_final_clip.mp4")
        final_clip.write_videofile(final_clip_path, codec='libx264', audio_codec='aac')
        cleanup_temp_files()

        # Calculate the total sum of sentiment scores
        total_score = sum(sentiments.values())
        if total_score == 0:
            sentiment_display = "\n".join(f"**{k}:** 0%" for k in sentiments)  # Handle case where all scores are zero
        else:
            sentiment_display = "\n".join(f"**{k}:** {v / total_score * 100:.1f}%" for k, v in sentiments.items())

        return final_clip_path, final_clip_path, sentiment_display
    else:
        return "No matching scene found", None, "No data"

        

# Custom CSS for additional styling
css = """
body {
    background-color: #ffffff;
    background-image: radial-gradient(#eb5726 1px, transparent 1px);
    background-size: 10px 10px;
    background-repeat: repeat;
    background-attachment: fixed;
}
#video_url {
    background-color: #ffffff;
    color: #282828;
    border: 2px solid #eb5726;
}
#description {
    background-color: #ffffff;
    color: #282828;
    border: 2px solid #eb5726;
}
#submit_button {
    background-color: #eb5726;
    color: #ffffff;
    border: 2px solid #ffffff;
}
#submit_button:hover {
    background-color: #f5986e;
    color: #ffffff;
    border: 2px solid #ffffff;
}
label[for="video_url"] {
    color: #eb5726 !important;
}
label[for="description"] {
    color: #eb5726 !important;
}
h3 {
    color: #eb5726;
}
.centered-markdown {
    text-align: center;
    background-color: #ffffff;
    padding: 10px;
}
#sickstadium-title {
    font-size: 3em !important;
    font-weight: bold;
    text-transform: uppercase;
}
"""

def save_uploaded_file(uploaded_file):
    upload_dir = "uploaded_videos"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{uuid.uuid4()}.mp4")
    with open(file_path, "wb") as f:
        f.write(uploaded_file)
    return file_path

with gr.Blocks(theme=custom_theme, css=css) as demo:
    with gr.Column():
        gr.Markdown("# **Sickstadium AI**")
        video_url = gr.Textbox(label="Video URL:")
        video_file = gr.File(label="Upload Video File:", type="binary")
        description = gr.Textbox(label="Describe your clip:")
        submit_button = gr.Button("Process Video")
        video_output = gr.Video(label="Processed Video")
        download_output = gr.File(label="Download Processed Video")
        sentiment_output = gr.Markdown(label="Sentiment Scores")  # Using Markdown to display sentiment scores
        submit_button.click(fn=display_results, inputs=[video_url, video_file, description], outputs=[video_output, download_output, sentiment_output])

demo.launch()