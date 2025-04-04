import argostranslate.package
import argostranslate.translate

# Скачиваем список доступных пакетов
packages = argostranslate.package.get_available_packages()

# Фильтруем нужную языковую пару
package_to_install = next(
    (p for p in packages if p.from_code == "ru" and p.to_code == "en"), None
)

# Скачиваем и устанавливаем
if package_to_install:
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
