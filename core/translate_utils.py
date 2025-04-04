from core.settings import TRANSLATOR, SOURCE_LANG, TARGET_LANG

def translate_text(text: str, source_lang: str = SOURCE_LANG, target_lang: str = TARGET_LANG) -> str:
    if not text.strip() or TRANSLATOR == "none":
        return text

    try:
        if TRANSLATOR == "argos":
            from argostranslate import translate as argos
            return argos.translate(text, source_lang, target_lang)

        elif TRANSLATOR == "deep":
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

        else:
            print(f"[Переводчик] Неизвестный TRANSLATOR: {TRANSLATOR}")
            return text

    except Exception as e:
        print(f"[Перевод] Ошибка ({TRANSLATOR}): {e}")
        return text