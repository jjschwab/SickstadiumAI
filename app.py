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
            fonts.GoogleFont("Roboto"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Roboto Mono"),
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
            body_background_fill="radial-gradient(circle at center, rgba(235, 87, 38, 1) 0%, rgba(235, 87, 38, 0) 70%), #ffffff",
            body_text_color="#282828",
            block_background_fill="#ffffff",
            block_title_text_color="#eb5726",
            block_label_text_color="#eb5726",
            button_primary_background_fill="#eb5726",
            button_primary_background_fill_hover="#ffffff",
            button_primary_text_color="#ffffff",
            button_primary_text_color_hover="#eb5726",
        )

custom_theme = CustomTheme()

def display_results(video_url, description):
    final_clip_path = process_video(video_url, description)
    if final_clip_path:
        return final_clip_path, final_clip_path
    return "No matching scene found", None

# Custom CSS for additional styling
css = """
body {
    background-color: #ffffff;
    background-image:
        radial-gradient(#eb5726 1px, transparent 1px),
        radial-gradient(circle at center, rgba(235, 87, 38, 1) 0%, rgba(235, 87, 38, 0) 70%);
    background-size: 10px 10px, cover;
    background-repeat: repeat, no-repeat;
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
    font-size: 2em !important;
    font-weight: bold;
}
"""

with gr.Blocks(theme=custom_theme, css=css) as demo:
    with gr.Column():
        gr.Markdown("# **Sickstadium AI**", elem_classes="centered-markdown", elem_id="sickstadium-title")
        gr.Markdown("### Upload your videos. Find sick clips. Tell your truth.", elem_classes="centered-markdown")
        gr.Markdown("**Welcome to Sickstadium AI. Our goal is to empower content creators with the ability to tell their stories without the friction of traditional video editing software. Skip the timeline, and don't worry about your experience in video editing. Upload your video, explain the clip you want, and let our AI video editor do the work for you. Get more info about the Sickstadium project at [Strongholdlabs.io](https://strongholdlabs.io/)**", elem_classes="centered-markdown")
        video_url = gr.Textbox(label="Video URL:", elem_id="video_url")
        description = gr.Textbox(label="Describe your clip:", elem_id="description")
        submit_button = gr.Button("Process Video", elem_id="submit_button")
        video_output = gr.Video(label="Processed Video", elem_id="video_output")
        download_output = gr.File(label="Download Processed Video", elem_id="download_output")
        submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])

demo.launch()
