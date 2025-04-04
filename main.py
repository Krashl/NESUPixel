import gradio as gr
from modes.txt2img import create_txt2img_ui
from core.settings import GRADIO_HOST, GRADIO_PORT


def build_interface():
    with gr.Blocks(title="NESUPixel") as app:
        gr.Markdown("# NESUPixel \U0001F4F7 - Text-to-Image Generator")
        create_txt2img_ui()
    return app


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False)