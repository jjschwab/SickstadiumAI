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
h1 {
    color: #282828;
    text-align: center;
}
h2 {
    color: #eb5726;
}
h3 {
    color: #ffffff;
} 
h4 {
    color: #eb5726;
} 
h5 {
    color: #282828;
}
h6 {
    color: #eb5726; 
}
#video_url {
    background-color: #eb5726;
    color: #ffffff;
    border: 2px solid #ecf0f1;
}
#description {
    background-color: #ecf0f1;
    color: #eb5726;
    border: 2px solid #eb5726;
}
#submit_button {
    background-color: #ffffff;
    color: #F06230;
    border: 2px solid #eb5726;
}
#submit_button:hover {
    background-color: #c0392b;
}
#video_output, #download_output {
    border: 1px solid #eb5726;
}
.centered-markdown {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    width: 100%;
}
"""

js = """
document.querySelector('label[for="video_url"]').style.color = '#ffffff';
document.querySelector('label[for="description"]').style.color = '#eb5726';
"""

with gr.Blocks(css=css) as demo:
    with gr.Column():
        gr.Markdown("# Sickstadium AI", elem_classes="centered-markdown")
        gr.Markdown("### This is a brief description for the webpage.", elem_classes="centered-markdown")
        video_url = gr.Textbox(info="Video URL or Filepath", elem_id="video_url")
        description = gr.Textbox(label="Description of desired clip", elem_id="description")
        submit_button = gr.Button("Process Video", elem_id="submit_button")
        video_output = gr.Video(label="Processed Video", elem_id="video_output")
        download_output = gr.File(label="Download Processed Video", elem_id="download_output")
        submit_button.click(fn=display_results, inputs=[video_url, description], outputs=[video_output, download_output])
        gr.HTML(f"<script>{js}</script>")

demo.launch()
