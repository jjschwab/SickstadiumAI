import gradio as gr
from video_processing import process_video

def display_results(video_url, description):
    final_clip_path = process_video(video_url, description)
    return final_clip_path, final_clip_path

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# My AI Video Processing App")
    video_url = gr.Textbox(label="Video URL or Filepath")
    description = gr.Textbox(label="Description of desired clip")
    video_output = gr.Video(label="Processed Video")
    download_output = gr.File(label="Download Processed Video")
    submit_button = gr.Button("Process Video")
    submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])

# Launch the app
demo.launch()

