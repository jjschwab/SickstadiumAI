import os
import gradio as gr
from video_processing import process_video
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
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
    if uploaded_file is None:
        return None  # Handle cases where no file was uploaded

    print(f"Received object type: {type(uploaded_file)}")  # Debug: Check the object type
    print(f"Uploaded file content: {uploaded_file}")  # Debug: Inspect the content

    # Handling file content based on its type
    if isinstance(uploaded_file, tuple):
        # If it's a tuple, it usually contains (filename, filedata)
        filename, filedata = uploaded_file
        file_path = os.path.join("uploaded_videos", filename)
        with open(file_path, "wb") as f:
            f.write(filedata)
    elif isinstance(uploaded_file, str):
        # If it's a string, assuming it's a file path
        file_path = uploaded_file
    else:
        raise ValueError("Unexpected file input type")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"File saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
    return file_path


def display_results(video_url, video_file, description):
    final_clip_path = None

    if video_url:
        final_clip_path = process_video(video_url, description, is_url=True)
    elif video_file:
        video_file_path = save_uploaded_file(video_file)
        if video_file_path:
            final_clip_path = process_video(video_file_path, description, is_url=False)
        else:
            return "No file provided or file save error", None

    if final_clip_path:
        return final_clip_path, final_clip_path
    else:
        return "No matching scene found", None

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

with gr.Blocks() as demo:
    with gr.Column():
        video_url = gr.Textbox(label="Video URL")
        video_file = gr.File(label="Upload Video File", type="file")
        description = gr.Textbox(label="Describe your clip")
        submit_button = gr.Button("Process Video")
        video_output = gr.Video(label="Processed Video")
        download_output = gr.File(label="Download Processed Video")
        submit_button.click(
            fn=display_results,
            inputs=[video_url, video_file, description],
            outputs=[video_output, download_output]
        )

demo.launch()