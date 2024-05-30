import gradio as gr
from video_processing import process_video

def display_results(video_url, description):
    final_clip_path = process_video(video_url, description)
    if final_clip_path:
        return final_clip_path, final_clip_path
    return "No matching scene found", None

with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Sickstadium AI!")
    gr.Markdown("Enter the URL of a YouTube video:")
    video_url = gr.Textbox(label="Video URL or Filepath")
    gr.Markdown("Describe the content you want to clip:")
    description = gr.Textbox(label="Description of desired clip")
    submit_button = gr.Button("Process Video")
    video_output = gr.Video(label="Processed Video")
    download_output = gr.File(label="Download Processed Video")
    submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])

demo.launch()
