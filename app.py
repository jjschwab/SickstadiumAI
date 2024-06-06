import gradio as gr
from video_processing import process_video, download_video, find_scenes, analyze_scenes, extract_best_scene, cleanup_temp_files
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import uuid
import os
from typing import Iterable


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
    output_message = None
    if video_url:
        video_path = download_video(video_url)
    elif video_file:
        video_path = save_uploaded_file(video_file)
    else:
        output_message = "No video provided"

    if not output_message:
        scenes = find_scenes(video_path)
        if not scenes:
            output_message = "No scenes detected"

    if not output_message:
        best_scene_info = analyze_scenes(video_path, scenes, description)
        if best_scene_info:
            best_scene = best_scene_info[0] if isinstance(best_scene_info, tuple) else None
            sentiment_distribution = best_scene_info[-1] if isinstance(best_scene_info, tuple) else None
            final_clip = extract_best_scene(video_path, best_scene) if best_scene else None
            if final_clip:
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                final_clip_path = os.path.join(output_dir, f"{uuid.uuid4()}_final_clip.mp4")
                final_clip.write_videofile(final_clip_path, codec='libx264', audio_codec='aac')
                cleanup_temp_files()

                plot = create_radial_plot(sentiment_distribution) if sentiment_distribution else "No sentiment data available"
                return final_clip_path, plot
            else:
                output_message = "No matching scene found"
        else:
            output_message = "Analysis failed or no suitable scenes found"

    return None, output_message


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
label[for="video_url"], label[for="description"] {
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
with gr.Blocks(theme=custom_theme, css=css) as demo:
    with gr.Column():
        video_url_input = gr.Textbox(label="Video URL:", elem_id="video_url")
        video_file_input = gr.File(label="Upload Video File:", interactive=True, file_types=["video"], type="binary")
        description_input = gr.Textbox(label="Describe your clip:", elem_id="description")
        submit_button = gr.Button("Process Video", elem_id="submit_button")
        video_output = gr.Video(label="Processed Video", elem_id="video_output")
        sentiment_plot_output = gr.Plot(label="Sentiment Distribution", elem_id="sentiment_plot")

        submit_button.click(
            fn=display_results,
            inputs=[video_url_input, video_file_input, description_input],
            outputs=[video_output, sentiment_plot_output]
        )

demo.launch()
