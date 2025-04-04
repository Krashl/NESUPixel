# core/comfy_api.py

import json
import uuid
import requests
import websocket
import threading
import time

from core.settings import COMFY_API_URL, WORKFLOW_PATH, DEBUG


def generate_client_id() -> str:
    return str(uuid.uuid4())

def load_and_patch_workflow(workflow_name: str, node_overrides: dict) -> dict:
    workflow_path = WORKFLOW_PATH / f"{workflow_name}.json"
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    for node_id, inputs in node_overrides.items():
        if node_id in workflow:
            workflow[node_id]["inputs"].update(inputs)
    return workflow

def queue_prompt(workflow_dict: dict, client_id: str) -> str:
    response = requests.post(
        f"{COMFY_API_URL}/prompt",
        json={"prompt": workflow_dict, "client_id": client_id}
    )
    response.raise_for_status()
    return response.json().get("prompt_id")

def interrupt():
    try:
        response = requests.post(f"{COMFY_API_URL}/interrupt")
        return response.status_code == 200
    except Exception as e:
        print(f"[Interrupt] Ошибка при прерывании: {e}")
        return False

def get_node_outputs_from_history(prompt_id: str, node_ids: list[str]) -> dict[str, list[dict]]:
    """
    Получает все выходные данные указанных нод из истории выполнения запроса в ComfyUI.

    Как работает история в ComfyUI:
    ------------------
    После отправки воркфлоу (prompt) в ComfyUI через API, ему присваивается уникальный `prompt_id` (чаще всего это UUID).
    Этот `prompt_id` используется для получения истории выполнения, хранящейся на стороне Comfy-сервера.

    Структура запроса:
    -------------------
    GET /history/{prompt_id}

    Ответ сервера представляет собой словарь, где ключ — это сам `prompt_id`, а значение — структура с результатами, состоящая из:

    - "prompt": словарь всех нод и их параметров, отправленных в Comfy на исполнение.  
      Это полный JSON воркфлоу, включая заполненные значения, имена нод, связи между ними.

    - "outputs": результаты выполнения. Ключи — это ID нод, значения — словари с выходами нод.  
      Например, для SaveImage будет список картинок:
        {
          "16": {
              "images": [
                  {
                      "filename": "Txt2Img_00006_.png",
                      "subfolder": "NESUPixel\\Txt2Img",
                      "type": "output"
                  }
              ]
          }
        }

    - "status": содержит статус выполнения, полезен для логирования, диагностики и отслеживания хода генерации:
        {
            "status_str": "success",
            "completed": true,
            "messages": [
                ["execution_start", {...}],
                ["execution_cached", {...}],
                ["execution_success", {...}]
            ]
        }

    - "meta": содержит вспомогательные технические данные, такие как `display_node`, `real_node_id`, `parent_node`.  
      Может быть полезно для отображения дерева нод, трассировки и UI-визуализации в будущем.

    Аргументы:
    ----------
    prompt_id : str
        Уникальный идентификатор воркфлоу-запроса, полученный при отправке задачи на генерацию (`queue_prompt()`).
    
    node_ids : list[str]
        Список ID нод, выходные данные которых нужно извлечь (например: ["16", "13", "7"]).

    Возвращает:
    -----------
    dict[str, list[dict]]
        Словарь вида {node_id: list_outputs}, где list_outputs — список словарей, представляющих выход ноды.
        Это может быть:
        - изображения (images)
        - текст (text)
        - аудио (audio)
        - или любой другой формат, возвращённый нодой.

    Пример использования:
    ----------------------
    outputs = get_node_outputs_from_history("a1b2c3", ["16", "13"])
    images = outputs["16"]
    first_image_path = images[0]["filename"]

    Примечание:
    ------------
    Эта функция **универсальна** и может использоваться не только для изображений,
    но и для любых других типов данных, возвращаемых нодами ComfyUI.
    Она является рекомендованным способом получения результатов **в случае кэша** или
    отсутствия события через WebSocket.
    """

    try:
        response = requests.get(f"{COMFY_API_URL}/history/{prompt_id}")
        if response.status_code != 200:
            return {}

        data = response.json()
        if DEBUG:
             print(f"[History] {data}")

        prompt_data = data.get(prompt_id, {})
        outputs = prompt_data.get("outputs", {})

        result = {}

        for node_id in node_ids:
            if node_id in outputs:
                result[node_id] = outputs[node_id]

        return result

    except Exception as e:
        print(f"[History] ⚠ Ошибка при получении истории: {e}")
        return {}

def monitor_websocket_for_node_output(client_id: str, expected_prompt_id: str, watch_node_ids: list[str], timeout: int = 15) -> dict[str, list[dict]]:
    """
    Слушает WebSocket от ComfyUI и извлекает выходы из указанных нод по завершении генерации.

    Что делает:
    ------------
    1. Отправляем запрос на генерацию с `client_id`
    2. Слушаем WebSocket сервер ComfyUI по адресу `/ws?clientId=...`
    3. При получении события `executed` от нужной ноды — забираем ее output (images, text, etc)
    4. Если нужная нода не была выполнена — функция вернет пустой словарь

    Поддерживает случаи:
    ---------------------
    - Прямой вызов (neкэш)
    - Можно использовать как дополнение к get_node_outputs_from_history()

    Аргументы:
    ----------
    client_id : str
        Уникальный идентификатор клиента
    expected_prompt_id : str
        ID запроса генерации
    watch_node_ids : list[str]
        Список нод, которые мы ждём

    Возвращает:
    -----------
    dict[str, list[dict]]
        Словарь с результатами по нодам
    """
    result: dict[str, list[dict]] = {}
    event = threading.Event()

    def on_message(ws, message):
        try:
            data = json.loads(message)

            if DEBUG and data.get("type") != "crystools.monitor":
                print(f"[WS] 📩 {data}")

            if data.get("type") == "executed":
                d = data.get("data", {})
                if d.get("prompt_id") != expected_prompt_id:
                    return
                node_id = d.get("node")
                if node_id in watch_node_ids:
                    output_data = d.get("output", {})
                    result[node_id] = []
                    for key, value in output_data.items():
                        if isinstance(value, list):
                            result[node_id].extend(value)
                    event.set()
                    ws.close()
        except Exception as e:
            print(f"[WS] ⚠ Ошибка: {e}")

    ws_url = COMFY_API_URL.replace("http", "ws") + f"/ws?clientId={client_id}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)

    thread = threading.Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()

    success = event.wait(timeout)

    if ws.sock and ws.sock.connected:
        ws.close()

    return result if success else {}

# core/comfy_api.py (дополнение к существующему коду)

def monitor_until_nodes_ready(
    client_id: str,
    prompt_id: str,
    watch_node_ids: list[str],
    timeout: int = 15,
    check_interval: float = 2.0,
    interrupt_on_ready: bool = False,
    preview_callback = None,
) -> dict[str, list[dict]]:
    """
    Ожидает, пока указанные ноды появятся в истории выполнения ComfyUI.
    Закрывает WebSocket при достижении результата или по таймауту.

    Использует стратегию:
    - Слушает WebSocket события (в фоне)
    - Периодически запрашивает историю выполнения
    - Как только все ноды есть в истории -- завершает
    """
    result: dict[str, list[dict]] = {}
    event = threading.Event()
    received_executed = set()

    def request_interrupt():
        try:
            url = COMFY_API_URL + "/interrupt"
            response = requests.post(url)
            if DEBUG:
                print(f"[INTERRUPT] Запрос отправлен: {response.status_code}")
        except Exception as e:
            print(f"[INTERRUPT] ⚠ Ошибка при прерывании: {e}")

    def on_message(ws, message):
        nonlocal received_executed
        try:
            # Проверяем, является ли сообщение бинарным (превью)
            if isinstance(message, bytes):
                if preview_callback and callable(preview_callback):
                    try:
                        # Важно! Пропускаем первые 8 байт, которые являются заголовком
                        preview_data = message[8:]
                        
                        if DEBUG:
                            print(f"[WS] Получены бинарные данные (превью), размер: {len(message)} байт, обрабатываем {len(preview_data)} байт после пропуска заголовка")
                        
                        preview_callback(preview_data)
                    except Exception as e:
                        if DEBUG:
                            print(f"[WS] ⚠ Ошибка при обработке бинарных данных как превью: {e}")
                return
            
            # Если не бинарные данные, пробуем декодировать как JSON
            data = json.loads(message)

            if DEBUG and data.get("type") != "crystools.monitor":
                print(f"[WS] 📩 {data}")

            if data.get("type") == "executed":
                d = data.get("data", {})
                if d.get("prompt_id") == prompt_id and d.get("node") in watch_node_ids:
                    received_executed.add(d.get("node"))
                    if DEBUG:
                        print(f"[WS] ✅ Получено executed для ноды {d.get('node')}")
                    if all(n in received_executed for n in watch_node_ids):
                        if interrupt_on_ready:
                            request_interrupt()
                        event.set()
                        ws.close()

            elif data.get("type") == "execution_cached":
                d = data.get("data", {})
                if d.get("prompt_id") == prompt_id:
                    cached_nodes = d.get("nodes", [])
                    if all(n in cached_nodes for n in watch_node_ids):
                        if DEBUG:
                            print("[WS] 💾 Все нужные ноды закэшированы, завершаем.")
                        if interrupt_on_ready:
                            request_interrupt()
                        event.set()
                        ws.close()
                    else:
                        if DEBUG:
                            print("[WS] ⏳ Частично закэшировано -- продолжаем ожидание.")

            elif data.get("type") == "status":
                d = data.get("data", {})
                exec_info = d.get("status", {}).get("exec_info", {})
                if exec_info.get("queue_remaining") == 0:
                    if DEBUG:
                        print("[WS] ⏹ Очередь пуста, можно завершать.")
                    event.set()
                    ws.close()

            # Обработка специальных сообщений превью в JSON формате
            elif data.get("type") == "preview":
                if preview_callback and callable(preview_callback):
                    try:
                        preview_data = data.get("data", {}).get("image")
                        if preview_data:
                            import base64
                            # Конвертируем base64 в бинарные данные
                            binary_data = base64.b64decode(preview_data)
                            preview_callback(binary_data)
                            if DEBUG:
                                print(f"[WS] Получено JSON превью, размер: {len(binary_data)} байт")
                    except Exception as e:
                        if DEBUG:
                            print(f"[WS] ⚠ Ошибка при обработке JSON превью: {e}")

        except Exception as e:
            print(f"[WS] ⚠ Ошибка: {e}")

    # Запуск WebSocket
    ws_url = COMFY_API_URL.replace("http", "ws") + f"/ws?clientId={client_id}"
    ws = websocket.WebSocketApp(
        ws_url, 
        on_message=on_message,
        on_error=lambda ws, error: print(f"[WS] ⚠ Ошибка соединения: {error}"),
        on_close=lambda ws, code, reason: print("[WS] Соединение закрыто")
    )

    # Указываем параметр для получения бинарных данных
    def on_open(ws):
        if DEBUG:
            print("[WS] Соединение открыто, запрашиваем бинарные превью")
        # Отправляем запрос на получение превью в различных форматах для поддержки разных версий ComfyUI
        ws.send(json.dumps({"type": "subscribe", "data": {"channel": "preview"}}))
        ws.send(json.dumps({"type": "binary_preview"}))

    ws.on_open = on_open

    thread = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 5})
    thread.daemon = True
    thread.start()

    # Основной цикл ожидания (WebSocket или таймаут)
    start_time = time.time()
    success = event.wait(timeout)

    if ws.sock and ws.sock.connected:
        ws.close()

    # После того как generation завершена -- пробуем получить историю
    if DEBUG:
        print("[INFO] Запрашиваем историю после завершения...")

    result = get_node_outputs_from_history(prompt_id, watch_node_ids)
    return result if result else {}