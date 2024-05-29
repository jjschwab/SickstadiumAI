import gradio as gr
from video_processing import process_video

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# My AI Video Processing App")
    video_url = gr.Textbox(label="Video URL or Filepath")
    description = gr.Textbox(label="Description of desired clip")
    output = gr.Textbox(label="Output", interactive=False)
    submit_button = gr.Button("Process Video")
    submit_button.click(fn=process_video, inputs=[video_url, description], outputs=output)

# Launch the app
demo.launch()
