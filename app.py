import gradio as gr
from video_processing import process_video
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable

class SimpleTheme(Base):
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
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
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
            body_background_fill="#282828",
            body_text_color="#ffffff",
            block_background_fill="#ffffff",
            block_title_text_color="#282828",
            block_label_text_color="#eb5726",
            button_primary_background_fill="#eb5726",
            button_primary_background_fill_hover="#ffffff",
            button_primary_text_color="#ffffff",
            button_primary_text_color_hover="#eb5726",
        )

simple_theme = SimpleTheme()

def display_results(video_url, description):
    final_clip_path = process_video(video_url, description)
    if final_clip_path:
        return final_clip_path, final_clip_path
    return "No matching scene found", None

with gr.Blocks(theme=simple_theme) as demo:
    with gr.Column():
        gr.Markdown("# Sickstadium AI", elem_classes="centered-markdown")
        gr.Markdown("### This is a brief description for the webpage.", elem_classes="centered-markdown")
        video_url = gr.Textbox(label="Video URL or Filepath", elem_id="video_url")
        description = gr.Textbox(label="Description of desired clip", elem_id="description")
        submit_button = gr.Button("Process Video", elem_id="submit_button")
        video_output = gr.Video(label="Processed Video", elem_id="video_output")
        download_output = gr.File(label="Download Processed Video", elem_id="download_output")
        submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])

demo.launch()
