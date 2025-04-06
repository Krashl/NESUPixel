# modes/inpaint.py
import gradio as gr
import uuid
import time
import io
import threading
from PIL import Image, ImageOps
from pathlib import Path
from core.comfy_api import generate_client_id, queue_prompt, load_and_patch_workflow, monitor_until_nodes_ready, interrupt
from core.workflow_descriptor import WorkflowDescriptor
from core.translate_utils import translate_text
from core.settings import DEBUG, COMFY_API_URL, COMFY_OUTPUT_PATH, COMFY_INPUT_PATH, DEFAULT_SEED
from core.lora_utils import load_lora_presets

LORA_PRESETS = load_lora_presets()
LORA_ALIAS_MAP = {preset["alias"]: preset for preset in LORA_PRESETS}

def save_image_locally(image: Image.Image, prefix: str) -> str:
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

def generate_inpaint(
    positive_ru, negative_ru,
    editor_output: dict,
    _dup,
    invert_mask: bool,
    lora_name, lora_strength,
    seed, translate, use_prompt_assistant,
    preview_func=None
):
    client_id = generate_client_id()
    workflow = WorkflowDescriptor(
        name="inpaint",
        output_node_ids=["16"]
    )

    if DEBUG:
        print(f"[INPAINT] Данные редактора: {editor_output.keys()}")

    if translate:
        positive = translate_text(positive_ru)
        negative = translate_text(negative_ru) if negative_ru else ""
    else:
        positive = positive_ru
        negative = negative_ru or ""

    # Корректная обработка выходных данных ImageEditor
    base_image = editor_output["background"].convert("RGB")
    
    # Создаём правильную маску из слоёв
    if "layers" in editor_output and editor_output["layers"]:
        # Берем первый слой с маской (он содержит рисование пользователя)
        mask_layer = editor_output["layers"][0]
        
        if DEBUG:
            print(f"[INPAINT] Размер основного изображения: {base_image.size}")
            print(f"[INPAINT] Режим маски: {mask_layer.mode}")
            print(f"[INPAINT] Размер маски: {mask_layer.size}")
        
        # В слое RGBA, альфа канал содержит непрозрачность рисования
        # Этот канал и есть то, что нарисовал пользователь
        if mask_layer.mode == 'RGBA':
            # Извлекаем альфа-канал (это будет наша маска)
            r, g, b, alpha = mask_layer.split()
            
            # Alpha канал содержит непрозрачность: 0=прозрачно, 255=непрозрачно
            # Превращаем его в маску, где белый (255) = область для inpaint
            mask = alpha
            
            # Если пользователь нарисовал черным, то непрозрачные (255) части должны 
            # быть областями для inpaint, если не нужно инвертировать
            if not invert_mask:
                pass  # оставляем как есть - непрозрачное (255) значит "применить inpaint"
            else:
                # Если нужно инвертировать - то непрозрачные части (255) НЕ обрабатываются
                mask = ImageOps.invert(mask)
        else:
            # Для других режимов (например, RGB без альфа)
            mask_layer = mask_layer.convert("L")
            
            # Проверим, какой у нас вариант маски - это может быть черное на белом
            # или белое на черном, в зависимости от того, как пользователь рисовал
            
            # Посчитаем средний уровень яркости маски
            avg_brightness = sum(mask_layer.getdata()) / (mask_layer.width * mask_layer.height)
            
            if DEBUG:
                print(f"[INPAINT] Средняя яркость маски: {avg_brightness}")
            
            # Если маска преимущественно светлая, значит пользователь нарисовал темным
            # и мы хотим, чтобы темные участки (то что он нарисовал) были заменены
            if avg_brightness > 128:
                # Инвертируем, чтобы то, что нарисовал пользователь, стало белым
                mask = ImageOps.invert(mask_layer)
            else:
                # Если пользователь рисовал светлым на темном, оставляем как есть
                mask = mask_layer
            
            # Теперь применяем желание пользователя по инверсии
            if invert_mask:
                mask = ImageOps.invert(mask)
    else:
        # Если слоев нет, создаем пустую маску
        if DEBUG:
            print("[INPAINT] Слои не найдены, создаем пустую маску")
        mask = Image.new("L", base_image.size, 0)  # Черная маска (ничего не обрабатывается)
    
    if DEBUG:
        # Сохраним оба изображения локально для отладки
        debug_dir = Path("temp")
        debug_dir.mkdir(exist_ok=True)
        debug_base_path = debug_dir / f"base_{uuid.uuid4()}.png"
        debug_mask_path = debug_dir / f"mask_{uuid.uuid4()}.png"
        base_image.save(debug_base_path)
        mask.save(debug_mask_path)
        print(f"[INPAINT] Отладочные файлы сохранены: {debug_base_path}, {debug_mask_path}")

    # Сохраняем изображения в директорию входных данных ComfyUI
    rel_base_path = save_image_locally(base_image, "base")
    rel_mask_path = save_image_locally(mask, "mask")

    if DEBUG:
        print(f"[INPAINT] Пути для ComfyUI: base={rel_base_path}, mask={rel_mask_path}")

    patch = {
        "1": {"string": positive},
        "2": {"string": negative},
        "3": {"image": rel_base_path},
        "4": {"image": rel_mask_path},
        "6": {
            "lora_name": lora_name,
            "strength_model": lora_strength,
            "strength_clip": lora_strength
        } if lora_name else {
            "strength_model": 0.0,
            "strength_clip": 0.0
        },
        "8": {"seed": seed},
    }

    if DEBUG:
        print(f"[INPAINT] Параметры воркфлоу: {patch}")

    wf = load_and_patch_workflow(workflow.name, patch)
    prompt_id = queue_prompt(wf, client_id)

    def process_preview(preview_data):
        if preview_func:
            try:
                preview_image = Image.open(io.BytesIO(preview_data))
                if DEBUG:
                    print(f"[INPAINT] Получено превью размером {preview_image.size}")
                preview_func(preview_image)
            except Exception as e:
                if DEBUG:
                    print(f"[PREVIEW] ⚠ Ошибка при обработке превью: {e}")

    result = monitor_until_nodes_ready(
        client_id,
        prompt_id,
        workflow.output_node_ids,
        interrupt_on_ready=True,
        preview_callback=process_preview
    )

    if DEBUG:
        print(f"[INPAINT] Результат после monitor_until_nodes_ready: {result}")

    if not result:
        if DEBUG:
            print("[INPAINT] Ожидаем 2 секунды и запрашиваем историю снова")
        time.sleep(2)
        result = monitor_until_nodes_ready(client_id, prompt_id, workflow.output_node_ids, interrupt_on_ready=False)
        if DEBUG:
            print(f"[INPAINT] Результат после повторного запроса: {result}")

    # Обработка результата и загрузка изображения
    if result and "16" in result:
        images = result["16"].get("images", [])
        if images:
            file_info = images[0]
            filename = file_info['filename']
            subfolder = file_info.get('subfolder', '')
            url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}"
            
            # Корректируем путь к файлу (строковая операция, а не операция с Path)
            path_str = str(COMFY_OUTPUT_PATH)
            correct_path = Path(path_str) / subfolder / filename
            
            # Альтернативный путь (без дублирования NESUPixel)
            if "NESUPixel" in path_str:
                alt_path_str = path_str.replace("NESUPixel", "", 1)  # заменяем только первое вхождение
                alt_path = Path(alt_path_str) / subfolder / filename
            else:
                alt_path = correct_path
            
            if DEBUG:
                print(f"[INPAINT] Пробуем пути: {correct_path} или {alt_path}")
            
            # Третий вариант пути (исходя из логов)
            output_folder = Path("output") 
            third_path = output_folder / "NESUPixel" / subfolder / filename
            
            if DEBUG:
                print(f"[INPAINT] Также пробуем путь: {third_path}")
            
            # Пробуем все возможные пути
            if correct_path.exists():
                if DEBUG:
                    print(f"[INPAINT] Файл найден по основному пути: {correct_path}")
                return url, str(correct_path)
            elif alt_path.exists():
                if DEBUG:
                    print(f"[INPAINT] Файл найден по альтернативному пути: {alt_path}")
                return url, str(alt_path)
            elif third_path.exists():
                if DEBUG:
                    print(f"[INPAINT] Файл найден по третьему пути: {third_path}")
                return url, str(third_path)
            else:
                # Попробуем скачать файл через URL
                try:
                    import requests
                    if DEBUG:
                        print(f"[INPAINT] Пытаемся скачать через URL: {url}")
                    
                    response = requests.get(url)
                    if response.status_code == 200:
                        # Сохраняем в локальной директории temp
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        temp_file = temp_dir / filename
                        
                        with open(temp_file, "wb") as f:
                            f.write(response.content)
                        
                        if DEBUG:
                            print(f"[INPAINT] Файл скачан и сохранен: {temp_file}")
                        
                        return url, str(temp_file)
                except Exception as e:
                    if DEBUG:
                        print(f"[INPAINT] ⚠ Ошибка при скачивании: {e}")
                
                if DEBUG:
                    print(f"[INPAINT] ⚠ Файл не найден по путям, возвращаем только URL")
                # Если не удалось найти файл, возвращаем только URL
                return url, None
    
    if DEBUG:
        print("[INPAINT] ⚠ Не удалось получить результат")
    return "", None

def create_inpaint_ui():
    with gr.Tab("Inpaint (замена области)"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Позитивный промпт (на русском)")
                negative = gr.Textbox(label="Негативный промпт", placeholder="по желанию")
                translate = gr.Checkbox(label="Перевести промпт в английский", value=True)
                assistant = gr.Checkbox(label="Использовать помощника (в будущем)", visible=False)

                draw = gr.ImageEditor(
                    label="Изображение + маска (рисуйте поверх)",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    brush=gr.Brush(colors=["#000000"]),
                    eraser=gr.Eraser(),
                    transforms=["crop", "resize", "rotate", "flip"],
                    height=512,
                    width=512
                )

                invert_mask = gr.Checkbox(label="Инвертировать маску (по умолчанию черное заменяется)", value=False)

                with gr.Accordion("LoRA (дополнительно)", open=False):
                    lora_alias = gr.Dropdown(choices=[p["alias"] for p in LORA_PRESETS], label="Выбрать стиль", allow_custom_value=True)
                    lora_path = gr.Textbox(label="LoRA путь (авто из стиля)", interactive=True)
                    lora_strength = gr.Slider(0, 1, value=0.0, step=0.05, label="Сила LoRA")

                seed = gr.Number(label="Seed (-1 = случайный)", value=DEFAULT_SEED, precision=0)

                with gr.Row():
                    generate_btn = gr.Button("Сгенерировать")
                    interrupt_btn = gr.Button("Прервать генерацию", variant="stop")

            with gr.Column(scale=1):
                image_output = gr.Image(label="Результат", type="pil")
                status_text = gr.Textbox(label="Статус", value="Готов к генерации", interactive=False)
                download = gr.File(label="Скачать результат")

        def apply_lora_preset(alias):
            preset = LORA_ALIAS_MAP.get(alias)
            if not preset:
                return gr.update(), gr.update()
            return preset["filename"], preset.get("recommended_strength", 0.7)

        preview_images = []
        is_generating = threading.Event()

        def on_interrupt():
            if is_generating.is_set():
                success = interrupt()
                return gr.update(value="Прерывание отправлено..." if success else "Ошибка при прерывании")
            return gr.update(value="Нет активной генерации")

        def on_generate(*args):
            nonlocal preview_images
            is_generating.set()
            preview_images.clear()
            yield gr.update(value=None), gr.update(value="Начало генерации..."), gr.update(value=None)

            def preview_handler(img):
                nonlocal preview_images
                preview_images.append(img)
                if DEBUG:
                    print(f"[UI] Получено новое превью #{len(preview_images)}, размер: {img.size}")

            generation_result = [None, None]
            generation_complete = threading.Event()

            def run_generation():
                nonlocal generation_result
                try:
                    if DEBUG:
                        print("[UI] Запуск generate_inpaint")
                        print(f"[UI] Аргументы: {args}")
                    result = generate_inpaint(*args, preview_func=preview_handler)
                    generation_result[0] = result[0]  # url
                    generation_result[1] = result[1]  # file_path
                    if DEBUG:
                        print(f"[UI] Результат генерации: URL={result[0]}, Path={result[1]}")
                except Exception as e:
                    print(f"[GENERATION] Ошибка: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    generation_complete.set()
                    is_generating.clear()

            generation_thread = threading.Thread(target=run_generation, daemon=True)
            generation_thread.start()

            last_preview_count = 0
            poll_interval = 0.1
            max_polls = int(30 / poll_interval)  # 30 секунд максимум

            for _ in range(max_polls):
                if generation_complete.is_set():
                    break
                current_preview_count = len(preview_images)
                if current_preview_count > last_preview_count:
                    last_preview_count = current_preview_count
                    latest_preview = preview_images[-1]
                    if DEBUG:
                        print(f"[UI] Отображаем превью #{current_preview_count}")
                    yield gr.update(value=latest_preview), gr.update(value=f"Генерация... Превью {current_preview_count}"), gr.update(value=None)
                time.sleep(poll_interval)

            if not generation_complete.is_set():
                if DEBUG:
                    print("[UI] Ожидаем завершения генерации...")
                generation_complete.wait()

            url, file_path = generation_result
            if DEBUG:
                print(f"[UI] Завершение генерации: URL={url}, Path={file_path}")
            
            if url:
                # Если есть URL, но нет файла, пробуем скачать изображение
                if not file_path or not Path(file_path).exists():
                    try:
                        import requests
                        if DEBUG:
                            print(f"[UI] Скачиваем изображение по URL: {url}")
                        
                        response = requests.get(url)
                        if response.status_code == 200:
                            temp_dir = Path("temp")
                            temp_dir.mkdir(exist_ok=True)
                            temp_file = temp_dir / f"inpaint_{uuid.uuid4()}.png"
                            
                            with open(temp_file, "wb") as f:
                                f.write(response.content)
                            
                            if DEBUG:
                                print(f"[UI] Файл скачан: {temp_file}")
                            
                            file_path = str(temp_file)
                    except Exception as e:
                        if DEBUG:
                            print(f"[UI] Ошибка при скачивании: {e}")
                
                # Пробуем отображать изображение через путь к файлу, а не URL
                if file_path and Path(file_path).exists():
                    # Читаем изображение с диска
                    try:
                        img = Image.open(file_path)
                        if DEBUG:
                            print(f"[UI] Загружено изображение из файла: {file_path}, размер: {img.size}")
                        
                        # Показываем изображение и предоставляем ссылку для скачивания
                        yield gr.update(value=img), gr.update(value="Генерация завершена!"), file_path
                        return
                    except Exception as e:
                        if DEBUG:
                            print(f"[UI] Ошибка при загрузке файла: {e}")
                
                # Если не удалось загрузить файл, пробуем URL
                try:
                    if DEBUG:
                        print(f"[UI] Пытаемся отобразить через URL: {url}")
                    yield gr.update(value=url), gr.update(value="Генерация завершена!"), file_path
                except Exception as e:
                    if DEBUG:
                        print(f"[UI] Ошибка при отображении URL: {e}")
                    # Пробуем отобразить последнее превью
                    if preview_images:
                        if DEBUG:
                            print(f"[UI] Отображаем последнее превью вместо результата")
                        yield gr.update(value=preview_images[-1]), gr.update(value="⚠️ Проблема с отображением результата. Используйте ссылку для скачивания."), file_path
                    else:
                        yield gr.update(value=None), gr.update(value="⚠️ Не удалось отобразить результат"), file_path
            else:
                yield gr.update(value=None), gr.update(value="Ошибка генерации или прервано"), None

        lora_alias.change(apply_lora_preset, inputs=[lora_alias], outputs=[lora_path, lora_strength])

        generate_btn.click(
            on_generate,
            inputs=[prompt, negative, draw, draw, invert_mask, lora_path, lora_strength, seed, translate, assistant],
            outputs=[image_output, status_text, download]
        )

        interrupt_btn.click(on_interrupt, inputs=[], outputs=[status_text])