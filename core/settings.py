from pathlib import Path

DEBUG = True

# === Пути ===
COMFY_BASE_PATH = Path("E:/AISoftware/ComfyUI/ComfyUI")
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows"
COMFY_OUTPUT_PATH = COMFY_BASE_PATH / "output" / "NESUPixel"
COMFY_INPUT_PATH = COMFY_BASE_PATH / "input" / "NESUPixel"
BASE_DIR = Path(__file__).resolve().parent.parent

# === Сеть ===
COMFY_API_URL = "http://127.0.0.1:8188"
GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860

# === Общие настройки генерации ===
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NEGATIVE_PROMPT = "blurry, bad anatomy, low quality"
DEFAULT_SEED = 0

# === Переводчики ===
# Переводчик: "argos", "deep", "none"
TRANSLATOR = "deep"
# Языковые коды
SOURCE_LANG = "ru"
TARGET_LANG = "en"

