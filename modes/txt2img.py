# modes/txt2img.py 
import gradio as gr
import uuid
import time
import io
import requests
import threading
from PIL import Image
from core.comfy_api import generate_client_id, queue_prompt, load_and_patch_workflow, monitor_until_nodes_ready, interrupt
from core.workflow_descriptor import WorkflowDescriptor
from core.translate_utils import translate_text
from core.settings import DEBUG, COMFY_API_URL, COMFY_OUTPUT_PATH, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_NEGATIVE_PROMPT, DEFAULT_SEED
from core.lora_utils import load_lora_presets
from pathlib import Path

# Глобально кэшируем пресеты
LORA_PRESETS = load_lora_presets()
LORA_ALIAS_MAP = {preset["alias"]: preset for preset in LORA_PRESETS}

def generate_txt2img(
    positive_ru, negative_ru, width, height,
    lora_name, lora_strength,
    seed, translate, use_prompt_assistant,
    preview_func=None  # Функция обратного вызова для превью
):
    client_id = generate_client_id()
    workflow = WorkflowDescriptor(
        name="txt2img",
        output_node_ids=["16"]
    )

    if translate:
        positive = translate_text(positive_ru)
        negative = translate_text(negative_ru) if negative_ru else ""
    else:
        positive = positive_ru
        negative = negative_ru or ""

    if DEBUG:
        print(f"[PROMPT] +: {positive}\n[PROMPT] -: {negative}")

    patch = {
        "1": {"string": positive},
        "2": {"string": negative},
        "3": {"int": width},
        "4": {"int": height},
        "8": {"seed": seed},
    }

    if lora_name:
        patch["6"] = {
            "lora_name": lora_name,
            "strength_model": lora_strength,
            "strength_clip": lora_strength
        }
    else:
        patch["6"] = {
            "strength_model": 0.0,
            "strength_clip": 0.0
        }

    wf = load_and_patch_workflow(workflow.name, patch)

    prompt_id = queue_prompt(wf, client_id)
    
    # Функция для обработки бинарных превью
    def process_preview(preview_data):
        if preview_func:
            try:
                # Открываем изображение из бинарных данных
                preview_image = Image.open(io.BytesIO(preview_data))
                if DEBUG:
                    print(f"[PREVIEW] Успешно получено превью: {preview_image.size}")
                preview_func(preview_image)
            except Exception as e:
                if DEBUG:
                    print(f"[PREVIEW] ⚠ Ошибка при обработке превью: {e}")
    
    # Передаем функцию обработки превью
    result = monitor_until_nodes_ready(
        client_id, 
        prompt_id, 
        workflow.output_node_ids, 
        interrupt_on_ready=True,
        preview_callback=process_preview  # Передаем функцию обработки превью
    )

    # Подождать немного перед запросом истории (например, 2 секунды)
    if not result:
        if DEBUG:
            print("[WAIT] ⏳ Подождём перед запросом истории...")
        time.sleep(2)
        result = monitor_until_nodes_ready(client_id, prompt_id, workflow.output_node_ids, interrupt_on_ready=False)

    if result and "16" in result:
        images = result["16"].get("images", [])
        if images:
            file_info = images[0]
            filename = file_info['filename']
            subfolder = file_info.get('subfolder', '')
            
            # Формируем URL для просмотра
            url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}"
            
            # Формируем путь для скачивания (может потребоваться настройка)
            # Получаем полное имя файла
            file_path = Path(COMFY_OUTPUT_PATH) / subfolder / filename if COMFY_OUTPUT_PATH else None
            if file_path and file_path.exists():
                return url, str(file_path)
            else:
                # Если файл не найден локально, пробуем скачать его
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        temp_file = temp_dir / filename
                        with open(temp_file, "wb") as f:
                            f.write(response.content)
                        return url, str(temp_file)
                except Exception as e:
                    print(f"Ошибка при скачивании файла: {e}")
                    
            # Если не удалось получить файл, возвращаем только URL
            return url, None
    return "", None


def create_txt2img_ui():
    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Позитивный промпт (на русском)")
                negative = gr.Textbox(label="Негативный промпт", placeholder="по желанию")
                translate = gr.Checkbox(label="Перевести промпт в английский", value=True)
                assistant = gr.Checkbox(label="Использовать помощника (в будущем)", visible=False)

                with gr.Row():
                    width = gr.Slider(512, 1600, value=DEFAULT_WIDTH, step=64, label="Ширина")
                    height = gr.Slider(512, 1600, value=DEFAULT_HEIGHT, step=64, label="Высота")

                with gr.Accordion("LoRA (дополнительно)", open=False):
                    lora_alias = gr.Dropdown(choices=[p["alias"] for p in LORA_PRESETS], label="Выбрать стиль", allow_custom_value=True)
                    lora_path = gr.Textbox(label="LoRA путь (авто из стиля)", interactive=True)
                    lora_strength = gr.Slider(0, 1, value=0.0, step=0.05, label="Сила LoRA")

                seed = gr.Number(label="Seed (-1 = случайный)", value=DEFAULT_SEED, precision=0)
                
                with gr.Row():
                    generate_btn = gr.Button("Сгенерировать")
                    interrupt_btn = gr.Button("Прервать генерацию", variant="stop")

            with gr.Column(scale=1):
                # Теперь используем только один компонент изображения
                image_output = gr.Image(label="Результат", type="pil")
                status_text = gr.Textbox(label="Статус", value="Готов к генерации", interactive=False)
                download = gr.File(label="Скачать результат")

        def apply_lora_preset(alias):
            preset = LORA_ALIAS_MAP.get(alias)
            if not preset:
                return gr.update(), gr.update()
            return preset["filename"], preset.get("recommended_strength", 0.7)

        # Глобальный список для хранения превью между вызовами
        # (нужно для совместного использования между генератором и функцией обратного вызова)
        preview_images = []
        # Переменная для отслеживания состояния генерации
        is_generating = threading.Event()

        def on_interrupt():
            if is_generating.is_set():
                success = interrupt()
                if DEBUG:
                    print(f"[INTERRUPT] Прерывание отправлено: {'успешно' if success else 'ошибка'}")
                return gr.update(value="Прерывание отправлено...")
            return gr.update(value="Нет активной генерации")

        def on_generate(*args):
            nonlocal preview_images
            # Устанавливаем флаг активной генерации
            is_generating.set()
            
            # Очищаем список превью перед новой генерацией
            preview_images.clear()
            
            # Очищаем изображение и устанавливаем статус
            yield gr.update(value=None), gr.update(value="Начало генерации..."), gr.update(value=None)
            
            # Функция обратного вызова для получения превью
            def preview_handler(img):
                nonlocal preview_images
                preview_images.append(img)
                if DEBUG:
                    print(f"[UI] Получено новое превью #{len(preview_images)}, размер: {img.size}")
            
            # Запускаем генерацию с функцией обновления превью
            generation_result = [None, None]  # [url, file_path]
            generation_complete = threading.Event()
            
            # Запускаем генерацию в отдельном потоке
            def run_generation():
                nonlocal generation_result
                try:
                    result = generate_txt2img(*args, preview_func=preview_handler)
                    generation_result[0] = result[0]  # url
                    generation_result[1] = result[1]  # file_path
                except Exception as e:
                    print(f"[GENERATION] Ошибка: {e}")
                finally:
                    generation_complete.set()
                    is_generating.clear()  # Сбрасываем флаг генерации
            
            generation_thread = threading.Thread(target=run_generation)
            generation_thread.daemon = True
            generation_thread.start()
            
            # Показываем превью в процессе генерации
            last_preview_count = 0
            poll_interval = 0.1  # Интервал проверки в секундах
            max_polls = int(30 / poll_interval)  # Максимальное время ожидания 30 секунд
            
            for _ in range(max_polls):
                if generation_complete.is_set():
                    break
                    
                # Проверяем, есть ли новые превью
                current_preview_count = len(preview_images)
                if current_preview_count > last_preview_count:
                    if DEBUG:
                        print(f"[UI] Найдены новые превью: было {last_preview_count}, стало {current_preview_count}")
                    
                    # Есть новое превью, показываем его
                    last_preview_count = current_preview_count
                    latest_preview = preview_images[-1]
                    yield gr.update(value=latest_preview), gr.update(value=f"Генерация... Получено превью {current_preview_count}"), gr.update(value=None)
                
                # Ждем немного перед следующей проверкой
                time.sleep(poll_interval)
            
            # Ждем завершения генерации, если еще не завершена
            if not generation_complete.is_set():
                generation_complete.wait()
            
            # Показываем окончательный результат
            url, file_path = generation_result
            if url:
                if DEBUG:
                    print(f"[UI] Генерация завершена. URL: {url}, Path: {file_path}")
                yield gr.update(value=url), gr.update(value="Генерация завершена!"), file_path
            else:
                if DEBUG:
                    print("[UI] Генерация завершена без результата")
                yield gr.update(value=None), gr.update(value="Ошибка генерации или прервано"), None

        lora_alias.change(
            fn=apply_lora_preset,
            inputs=[lora_alias],
            outputs=[lora_path, lora_strength]
        )

        # Обновляем обработчик кнопки для поддержки потока обновлений
        generate_btn.click(
            on_generate,
            inputs=[prompt, negative, width, height, lora_path, lora_strength, seed, translate, assistant],
            outputs=[image_output, status_text, download]
        )
        
        # Добавляем обработчик для кнопки прерывания
        interrupt_btn.click(
            on_interrupt,
            inputs=[],
            outputs=[status_text]
        )