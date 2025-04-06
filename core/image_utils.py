# core/image_utils.py
import io
import uuid
import requests
from typing import Tuple, Optional, Union, Dict, List
from pathlib import Path
from PIL import Image, ImageOps

from core.settings import DEBUG, COMFY_INPUT_PATH, COMFY_OUTPUT_PATH, COMFY_API_URL

def save_image_locally(image: Image.Image, prefix: str) -> str:
    """
    Сохраняет изображение в директории ввода ComfyUI и возвращает относительный путь.
    
    Args:
        image: Изображение PIL для сохранения
        prefix: Префикс имени файла
        
    Returns:
        Относительный путь в формате, понятном для ComfyUI
    """
    filename = f"{prefix}_{uuid.uuid4()}.png"
    relative_path = Path("NESUPixel") / filename
    abs_path = COMFY_INPUT_PATH / filename
    
    # Убедимся, что директория существует
    abs_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Сохраняем как PNG, явно указываем формат
    image.save(abs_path, format="PNG")
    
    if DEBUG:
        print(f"[SAVE] Изображение сохранено: {abs_path}")
    
    # Возвращаем относительный путь в формате, совместимом с ComfyUI
    return str(relative_path).replace("\\", "/")

def save_debug_image(image: Image.Image, prefix: str) -> Path:
    """
    Сохраняет изображение для отладки в директории debug.
    
    Args:
        image: Изображение PIL для сохранения
        prefix: Префикс имени файла
        
    Returns:
        Путь к сохраненному файлу
    """
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    debug_path = debug_dir / f"{prefix}_{uuid.uuid4()}.png"
    image.save(debug_path, format="PNG")
    
    if DEBUG:
        print(f"[DEBUG] Изображение сохранено: {debug_path}")
        
    return debug_path

def process_editor_output(editor_output: dict, invert_mask: bool = False) -> Tuple[Image.Image, Image.Image]:
    """
    Обрабатывает выход компонента gr.ImageEditor и возвращает базовое изображение и маску.
    
    Args:
        editor_output: Выходные данные от ImageEditor
        invert_mask: Нужно ли инвертировать маску
        
    Returns:
        Кортеж (базовое_изображение, маска)
    """
    if DEBUG:
        print(f"[EDITOR] Данные редактора: {editor_output.keys()}")
        
    # Получаем основное изображение
    base_image = editor_output["background"].convert("RGB")
    
    # Создаём маску из слоёв
    if "layers" in editor_output and editor_output["layers"]:
        # Берем первый слой с маской (он содержит рисование пользователя)
        mask_layer = editor_output["layers"][0]
        
        if DEBUG:
            print(f"[EDITOR] Режим маски: {mask_layer.mode}")
            print(f"[EDITOR] Размеры: база={base_image.size}, маска={mask_layer.size}")
        
        # В слое RGBA, альфа канал содержит непрозрачность рисования
        if mask_layer.mode == 'RGBA':
            # Извлекаем альфа-канал (это будет наша маска)
            r, g, b, alpha = mask_layer.split()
            
            # Alpha канал: 0=прозрачно, 255=непрозрачно
            mask = alpha
            
            # Применяем инверсию при необходимости
            if invert_mask:
                mask = ImageOps.invert(mask)
        else:
            # Для других режимов
            mask_layer = mask_layer.convert("L")
            
            # Анализируем яркость маски
            avg_brightness = sum(mask_layer.getdata()) / (mask_layer.width * mask_layer.height)
            
            if DEBUG:
                print(f"[EDITOR] Средняя яркость маски: {avg_brightness}")
            
            # Если маска светлая, пользователь рисовал темным
            if avg_brightness > 128:
                mask = ImageOps.invert(mask_layer)
            else:
                mask = mask_layer
            
            if invert_mask:
                mask = ImageOps.invert(mask)
    else:
        # Если слоев нет, создаем пустую маску
        if DEBUG:
            print("[EDITOR] Слои не найдены, создаем пустую маску")
        mask = Image.new("L", base_image.size, 0)  # Черная маска
        
    # Сохраняем отладочные изображения
    if DEBUG:
        save_debug_image(base_image, "base")
        save_debug_image(mask, "mask")
        
    return base_image, mask

def find_output_image(filename: str, subfolder: str) -> Optional[Path]:
    """
    Находит выходное изображение по возможным путям.
    
    Args:
        filename: Имя файла
        subfolder: Поддиректория
        
    Returns:
        Путь к найденному файлу или None
    """
    # Пробуем различные варианты путей
    possible_paths = []
    
    # Основной путь
    path_str = str(COMFY_OUTPUT_PATH)
    possible_paths.append(Path(path_str) / subfolder / filename)
    
    # Путь без дублирования NESUPixel
    if "NESUPixel" in path_str:
        alt_path_str = path_str.replace("NESUPixel", "", 1)
        possible_paths.append(Path(alt_path_str) / subfolder / filename)
    
    # Третий вариант пути
    possible_paths.append(Path("output") / "NESUPixel" / subfolder / filename)
    
    # Четвертый вариант
    possible_paths.append(Path("output") / subfolder / filename)
    
    if DEBUG:
        print(f"[FIND] Проверяем пути: {possible_paths}")
    
    # Пробуем все пути
    for path in possible_paths:
        if path.exists():
            if DEBUG:
                print(f"[FIND] Файл найден: {path}")
            return path
    
    return None

def download_image(url: str) -> Optional[Path]:
    """
    Скачивает изображение по URL и сохраняет во временной директории.
    
    Args:
        url: URL изображения
        
    Returns:
        Путь к скачанному файлу или None
    """
    try:
        if DEBUG:
            print(f"[DOWNLOAD] Скачивание: {url}")
        
        response = requests.get(url)
        if response.status_code == 200:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            filename = f"download_{uuid.uuid4()}.png"
            temp_file = temp_dir / filename
            
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            if DEBUG:
                print(f"[DOWNLOAD] Сохранено: {temp_file}")
            
            return temp_file
    except Exception as e:
        if DEBUG:
            print(f"[DOWNLOAD] Ошибка: {e}")
    
    return None

def get_output_image_info(result: Dict, node_id: str) -> Tuple[Optional[str], Optional[Path]]:
    """
    Извлекает информацию о выходном изображении из результата ComfyUI.
    
    Args:
        result: Результат выполнения ComfyUI
        node_id: ID ноды с выходным изображением
        
    Returns:
        Кортеж (url, путь_к_файлу)
    """
    if result and node_id in result:
        images = result[node_id].get("images", [])
        if images:
            file_info = images[0]
            filename = file_info['filename']
            subfolder = file_info.get('subfolder', '')
            url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}"
            
            # Пробуем найти файл
            file_path = find_output_image(filename, subfolder)
            
            if file_path:
                return url, file_path
            
            # Если файл не найден, пробуем скачать
            downloaded_file = download_image(url)
            if downloaded_file:
                return url, downloaded_file
            
            # Если не удалось скачать, возвращаем только URL
            return url, None
    
    return None, None