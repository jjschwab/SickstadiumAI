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
    background-color: #eb5726;
    color: #ffffff;
    font-family: 'Arial', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #282828;
}
#video_url {
    background-color: #eb5726;
    color: #ecf0f1;
    border: 1px solid #ecf0f1;
}
#description {
    background-color: #ecf0f1;
    color: #eb5726;
    border: 1px solid #eb5726;
}
#submit_button {
    background-color: #ffffff;
    color: #F06230;
    border: 1px solid #F06230;
}
#submit_button:hover {
    background-color: #c0392b;
}
#video_output, #download_output {
    border: 1px solid #eb5726;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Sickstadium AI")
    video_url = gr.Textbox(label="Video URL or Filepath", elem_id="video_url")
    description = gr.Textbox(label="Description of desired clip", elem_id="description")
    video_output = gr.Video(label="Processed Video", elem_id="video_output")
    download_output = gr.File(label="Download Processed Video", elem_id="download_output")
    submit_button = gr.Button("Process Video", elem_id="submit_button")
    submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])

demo.launch()
