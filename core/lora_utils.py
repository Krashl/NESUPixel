import json
from pathlib import Path

from core.settings import BASE_DIR

LORA_PRESET_PATH = BASE_DIR / "config" / "lora_presets.json"

def load_lora_presets() -> list[dict]:
    """
    Загружает список LoRA-пресетов из JSON-файла.
    Каждый пресет содержит: alias, filename, recommended_strength, keywords.
    """
    try:
        with open(LORA_PRESET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[LoRA] ⚠ Не удалось загрузить пресеты: {e}")
        return []
