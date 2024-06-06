import gradio as gr
import os

def save_and_display_video(video_file):
    if video_file is None:
        return None, "No video uploaded."

    try:
        if len(video_file) > 0:
            # Save the binary content to a file
            file_path = 'uploaded_video.mp4'  # Assuming .mp4 for simplicity
            with open(file_path, 'wb') as f:
                f.write(video_file)
            return file_path, "Video uploaded and displayed successfully."
        else:
            return None, "Uploaded file is empty."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

with gr.Blocks() as demo:
    with gr.Column():
        video_file = gr.UploadButton(label="Upload Video File", file_types=["mp4", "avi", "mov"], interactive=True)
        output_video = gr.Video()
        output_message = gr.Textbox(label="Output Message")
        submit_button = gr.Button("Display Video")
        submit_button.click(
            fn=save_and_display_video, 
            inputs=video_file, 
            outputs=[output_video, output_message]
        )

demo.launch()
