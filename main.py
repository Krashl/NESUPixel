# main.py
import gradio as gr
from modes.txt2img import create_txt2img_ui
from modes.inpaint import create_inpaint_ui  # ✅ новый импорт
from core.settings import GRADIO_HOST, GRADIO_PORT


def build_interface():
    with gr.Blocks(title="NESUPixel") as app:
        gr.Markdown("# NESUPixel 📷 - Image Generation Platform")
        create_txt2img_ui()
        create_inpaint_ui()  # ✅ добавляем вкладку inpaint
    return app


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False)
