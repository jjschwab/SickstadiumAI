import gradio as gr

def display_video(video_file):
    if video_file is None:
        return None, "No video uploaded."
    try:
        # Directly return the path to the uploaded video file for displaying
        return video_file, None
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

with gr.Blocks() as demo:
    with gr.Column():
        video_file = gr.UploadButton("Upload Video File", type="binary", file_types=["video"])
        output_video = gr.Video()
        output_message = gr.Textbox(label="Output Message")
        submit_button = gr.Button("Display Video")
        submit_button.click(
            fn=display_video, 
            inputs=video_file, 
            outputs=[output_video, output_message]
        )

demo.launch()
