import gradio as gr
from video_processing import process_video

def display_results(video_url, description):
    final_clip_path = process_video(video_url, description)
    if final_clip_path:
        return final_clip_path, final_clip_path
    return "No matching scene found", None

# Custom CSS
css = """
body {
    background-color: #2c3e50;
    color: #ecf0f1;
    font-family: 'Arial', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #ecf0f1;
}
#video_url, #description {
    background-color: #34495e;
    color: #ecf0f1;
    border: 1px solid #ecf0f1;
}
#submit_button {
    background-color: #e74c3c;
    color: #ecf0f1;
    border: 1px solid #ecf0f1;
}
#submit_button:hover {
    background-color: #c0392b;
}
#video_output, #download_output {
    border: 1px solid #ecf0f1;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# My AI Video Processing App")
    video_url = gr.Textbox(label="Video URL or Filepath", elem_id="video_url")
    description = gr.Textbox(label="Description of desired clip", elem_id="description")
    video_output = gr.Video(label="Processed Video", elem_id="video_output")
    download_output = gr.File(label="Download Processed Video", elem_id="download_output")
    submit_button = gr.Button("Process Video", elem_id="submit_button")
    submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])

demo.launch()
