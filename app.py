import gradio as gr
import os

def display_video(video_file):
    if video_file is None:
        return None, "No video uploaded."

    # Check if the uploaded file is a valid binary file
    try:
        if len(video_file) > 0:  # Simple check to confirm it contains content
            return video_file, None
        else:
            return None, "Uploaded file is empty."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

with gr.Blocks() as demo:
    with gr.Column():
        video_file = gr.File(label="Upload Video File", type="binary", file_types=["mp4", "avi", "mov"])
        output_video = gr.Video()
        output_message = gr.Textbox(label="Output Message")
        submit_button = gr.Button("Display Video")
        submit_button.click(
            fn=display_video, 
            inputs=video_file, 
            outputs=[output_video, output_message]
        )

demo.launch()
