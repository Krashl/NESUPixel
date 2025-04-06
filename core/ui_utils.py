# core/ui_utils.py
import io
import uuid
import time
import threading
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path
from PIL import Image
import gradio as gr

from core.settings import DEBUG
from core.comfy_api import interrupt

def apply_lora_preset(alias: str, preset_map: Dict) -> Tuple:
    """
    Применяет пресет LoRA на основе алиаса.
    
    Args:
        alias: Алиас пресета
        preset_map: Словарь с пресетами
        
    Returns:
        Кортеж (путь_к_файлу, рекомендуемая_сила)
    """
    preset = preset_map.get(alias)
    if not preset:
        return gr.update(), gr.update()
    return preset["filename"], preset.get("recommended_strength", 0.7)

def on_interrupt():
    """
    Обработчик прерывания генерации.
    
    Returns:
        Обновление для компонента статуса
    """
    success = interrupt()
    if DEBUG:
        print(f"[INTERRUPT] Прерывание отправлено: {'успешно' if success else 'ошибка'}")
    return gr.update(value="Прерывание отправлено..." if success else "Ошибка при прерывании")