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
    print(f"Received object type: {type(uploaded_file)}")
    if uploaded_file is None:
        return None  # Handle cases where no file was uploaded
    
    if isinstance(uploaded_file, gr.NamedString):
        print(f"File path from NamedString: {uploaded_file}")
        return uploaded_file  # Directly return the path if it's a NamedString

    upload_dir = "uploaded_videos"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)

    # Save the temporary file to a new location
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())  # Assuming file is a file-like object
        f.flush()
        os.fsync(f.fileno())  # Ensure all file data is flushed to disk

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

with gr.Blocks(theme=custom_theme, css=css) as demo:
    with gr.Column():
        gr.Markdown("# **Sickstadium AI**", elem_classes="centered-markdown", elem_id="sickstadium-title")
        gr.Markdown("### Upload your videos. Find sick clips. Tell your truth.", elem_classes="centered-markdown")
        gr.Markdown("**Welcome to Sickstadium AI. Our goal is to empower content creators with the ability to tell their stories without the friction of traditional video editing software. Skip the timeline, and don't worry about your video editing skills. Upload your video, describe the clip you want, and let our AI video editor do the work for you. Get more info about the Sickstadium project at [Strongholdlabs.io](https://strongholdlabs.io/)**", elem_classes="centered-markdown")
        video_url = gr.Textbox(label="Video URL:")
        video_file = gr.File(label="Upload Video File:")
        description = gr.Textbox(label="Describe your clip:")
        submit_button = gr.Button("Process Video")
        video_output = gr.Video(label="Processed Video")
        download_output = gr.File(label="Download Processed Video")
        submit_button.click(fn=display_results, inputs=[video_url, video_file, description], outputs=[video_output, download_output])

demo.launch()
