# FreeMoCap Enhanced v3.2 — Инструкция по установке

## Требования

- **Python 3.11** (рекомендуется)
- **Windows 10/11** (протестировано), Linux/macOS (не протестировано, но должно работать)
- **NVIDIA GPU** с поддержкой CUDA (рекомендуется для ускорения RTMPose)
- **Blender** (опционально, для 3D визуализации)

## Быстрый старт

### 1. Создайте виртуальное окружение

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### 2. Установите FreeMoCap

```bash
pip install freemocap
```

### 3. Установите дополнительные зависимости

```bash
pip install rtmlib onnxruntime opencv-contrib-python pyyaml
```

### 4. Скопируйте / клонируйте этот репозиторий

Скопируйте папку `freemocap_enhanced` в рабочую директорию или клонируйте:

```bash
git clone -b multi-actor-support <repo-url>
```

### 5. Запуск

```bash
cd freemocap_enhanced
python run_freemocap_enhanced.py
```

Приложение запустится с:
- Кастомным логотипом с авто-масштабированием
- Оригинальным диалогом приветствия FreeMoCap
- Enhanced экраном приветствия с переключателем RU/EN
- Всплывающим окном с заметками о последних обновлениях

## Что включено

### Улучшения пайплайна (v3.2)
- Фильтр дрожания (OneEuro + Butterworth)
- Уforcement длин костей (FK-BFS)
- Детекция и коррекция скольжения стоп
- Выравнивание плоскости пола (RANSAC + PCA)
- Детекция самоперекрытия
- Интерполяция NaN-пропусков
- Коррекция путаницы лево/право
- Логирование в реальном времени + экспорт в CSV

### Multi-Person трекинг (BETA)
- RTMDet + RTMPose для 2D детекции (133 ключевые точки)
- Эпиполярная геометрия + венгерский алгоритм
- Временной трекинг с глобальными ID
- Триангуляция DLT для каждого актёра
- Валидация физического взаимодействия
- ArUco маркерный fallback

### Улучшения интерфейса
- Выбор режима: Один актёр / Два актёра
- Двуязычный интерфейс (Русский / Английский)
- Кастомный логотип главного экрана (авто-масштаб)
- Настройки dual-actor с ArUco превью

## Запуск тестов

Поместите тестовые данные в `test_data/freemocap_test_data/` относительно корня проекта, или установите переменную окружения:

```bash
set FREEMOCAP_TEST_DATA=D:\path\to\test_data\freemocap_test_data
```

Затем запустите:

```bash
python -m pytest test_full_pipeline_integration.py -v
```

## Структура тестовых данных

```
test_data/
  freemocap_test_data/
    synchronized_videos/
      camera_0.mp4
      camera_1.mp4
      camera_2.mp4
    freemocap_test_data_camera_calibration.toml
```

## Решение проблем

- **Ошибки CUDA**: Убедитесь, что установлен `onnxruntime-gpu` вместо `onnxruntime`
- **Ошибки импорта**: Все зависимости должны быть установлены в одном виртуальном окружении
- **Логотип не отображается**: Файл `FREEMOCAP-BETA.png` должен находиться в той же папке, что и `run_freemocap_enhanced.py`
