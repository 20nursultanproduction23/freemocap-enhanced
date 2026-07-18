"""Append Section 13 to technical_report[RUS].docx"""
import docx

path = r"C:\freemocap-docs\technical_report[RUS].docx"
doc = docx.Document(path)

doc.add_page_break()

doc.add_heading("Раздел 13: Исследование — оценка адаптируемости внешних репозиториев", level=1)

doc.add_heading("13.1 Цель", level=2)
doc.add_paragraph(
    "До начала реализации multi-person tracking оценить три внешних репозитория "
    "multi-view оценки позы для определения, какой (если какой-либо) предоставляет "
    "переиспользуемые алгоритмы ассоциации людей между камерами. Ключевой вопрос: "
    "ассоциация — это обученная нейросеть, требующая весов re-identification, или "
    "чисто геометрический алгоритм, работающий с любым детектором?"
)

doc.add_heading("13.2 Оценённые репозитории", level=2)

# --- mvpose ---
doc.add_heading("13.2.1 mvpose (zju3dv)", level=3)
doc.add_paragraph("GitHub: zju3dv/mvpose")
doc.add_paragraph("Статья: Cross-view Parsing (ECCV 2018)")

p = doc.add_paragraph()
p.add_run("Алгоритм ассоциации: ").bold = True
p.add_run(
    "Двухэтапная система: (1) матрица сходства, объединяющая эпиполярную геометрию "
    "и ReID-признаки внешности, (2) SVT (Singular Value Thresholding) через ADMM "
    "полужёсткую релаксацию."
)

p = doc.add_paragraph()
p.add_run("Формат входных данных: ").bold = True
p.add_run(
    "COCO-17 ключевые точки (плоский список из 51 значения: 17 x [x,y,confidence]). "
    "Требуется pickle калибровки камер с P (3x4 проекция), K (3x3 внутренние), "
    "RT (3x4 внешние). Массив dimGroup задаёт количество людей на каждой камере."
)

p = doc.add_paragraph()
p.add_run("Ключевые функции: ").bold = True
p.add_run(
    "geometry_affinity() в src/m_utils/geometry.py:64-96 (эпиполярное расстояние → сходство), "
    "matchSVT() в src/models/matchSVT.py:16-94 (ADMM + SVD + двойная стохастическая "
    "проекция + транзитивное замыкание)."
)

p = doc.add_paragraph()
p.add_run("Зависимость от обученных весов: ").bold = True
p.add_run(
    "ГИБРИДНАЯ. ReID-компонент (ResNet-50, Market-1501, 1024-мерные эмбеддинги) требует "
    "предобученный checkpoint.pth.tar. ОДНАКО система имеет явный режим 'Geometry only' "
    "(model_config.py metric='Geometry only'), использующий ТОЛЬКО эпиполярное сходство — "
    "ноль обученных параметров. SVT-алгоритм — чистая математика."
)

p = doc.add_paragraph()
p.add_run("Адаптируемость к RTMPose 133 точкам: ").bold = True
p.add_run(
    "ВЫСОКАЯ. SVT-матчер (matchSVT.py) полностью универсален — принимает любую NxN матрицу "
    "сходства. Только функция geometry_affinity() хардкодит joint_num=17 в 3-4 местах. "
    "Маппинг RTMPose 133→COCO-17 — тривиальная индексация. К ядру сопоставления изменений не нужно."
)

# --- crossview ---
doc.add_heading("13.2.2 crossview_3d_pose_tracking (longcw)", level=3)
doc.add_paragraph("GitHub: longcw/crossview_3d_pose_tracking")
doc.add_paragraph("Статья: Cross-View Tracking (CVPR 2020)")

p = doc.add_paragraph()
p.add_run("Алгоритм ассоциации: ").bold = True
p.add_run(
    "ОТСУТСТВУЕТ в репозитории. README прямо указывает: 'Исходный код не включён — "
    "это коммерческий проект.' Репозиторий содержит только загрузчики данных, утилиты "
    "калибровки, код оценки (метрика PCP) и визуализацию."
)

p = doc.add_paragraph()
p.add_run("Зависимость от обученных весов: ").bold = True
p.add_run("N/A — ноль кода нейросетей, ноль весов, ноль ML-импортов во всём репозитории.")

p = doc.add_paragraph()
p.add_run("Адаптируемость: ").bold = True
p.add_run(
    "НУЛЕВАЯ для ассоциации. Утилиты калибровки/триангуляции (calib/calibration.py) "
    "переиспользуемы, но тривиальны — стандартный DLT."
)

# --- multiview_pose ---
doc.add_heading("13.2.3 multiview_pose (wusize)", level=3)
doc.add_paragraph("GitHub: wusize/multiview_pose")
doc.add_paragraph("Статья: Graph-based 3D Multi-Person Pose Estimation")

p = doc.add_paragraph()
p.add_run("Алгоритм ассоциации: ").bold = True
p.add_run(
    "Двухэтапный: (1) Эпиполярная геометрия через класс StereoGeometry (чистая математика — "
    "пересечение лучей + расстояние точки до эпиполярной линии), (2) GCN-уточнение "
    "(EdgeConvLayers) на основе CNN-признаков."
)

p = doc.add_paragraph()
p.add_run("Формат входных данных: ").bold = True
p.add_run(
    "НЕ ключевые точки — espera карты признаков CNN (NxVxCxHxW). "
    "Люди детектируются через специальный канал heatmap. Скелет Panoptic 15 точек, не COCO-17."
)

p = doc.add_paragraph()
p.add_run("Зависимость от обученных весов: ").bold = True
p.add_run(
    "GCN-скоринг требует обученных весов. ОДНАКО геометрический prior — "
    "exp(-coef × epipolar_distance) — чисто математический и даёт сильный начальный сигнал."
)

p = doc.add_paragraph()
p.add_run("Адаптируемость к RTMPose 133 точкам: ").bold = True
p.add_run(
    "СРЕДНЯЯ. Классы StereoGeometry/MonocularGeometry (utils.py) — чистые утилиты геометрии, "
    "независимые от формата ключевых точек. Но входной пайплайн ожидает карты признаков CNN."
)

doc.add_heading("13.3 Сводная таблица", level=2)

table = doc.add_table(rows=1, cols=5)
table.style = 'Table Grid'
hdr = table.rows[0].cells
for i, h in enumerate(["Критерий", "mvpose", "crossview_tracking", "multiview_pose", "Победитель"]):
    hdr[i].text = h

rows_data = [
    ("Код ассоциации", "Есть", "Нет (проприетарный)", "Есть", "mvpose / multiview"),
    ("Геометрическая ассоциация", "Эпиполяр + SVT", "N/A", "Эпиполяр + GCN", "mvpose (SVT)"),
    ("Без обученных весов", "Да (Geometry only)", "N/A", "Частично (geo prior)", "mvpose"),
    ("Формат входа → RTMPose", "COCO-17 (легко)", "N/A", "CNN-признаки (сложно)", "mvpose"),
    ("Уровень извлечения", "Низкий (3-4 значения)", "N/A", "Средний (обход CNN)", "mvpose"),
]

for row_data in rows_data:
    row = table.add_row().cells
    for i, val in enumerate(row_data):
        row[i].text = val

doc.add_heading("13.4 Вывод и рекомендация", level=2)

p = doc.add_paragraph()
p.add_run("Выбранный подход: собственная геометрическая ассоциация на основе mvpose + multiview_pose.\n\n").bold = True

doc.add_paragraph(
    "crossview_3d_pose_tracking отклонён — код ассоциации проприетарный."
)

doc.add_paragraph(
    "multiview_pose: утилиты геометрии (StereoGeometry, MonocularGeometry в utils.py) "
    "стоит извлечь как референсную реализацию. Однако пайплайн сопоставления требует "
    "CNN-карты признаков на входе — архитектурно несовместимо с RTMPose."
)

doc.add_paragraph(
    "mvpose — основная ссылка. Режим Geometry-only предоставляет именно то, что нужно: "
    "(1) эпиполярное сходство из пар ключевых точек камер, (2) SVT-алгоритм для "
    "многокамерного назначения. Единственная адаптация — маппинг индексов RTMPose 133→COCO-17."
)

doc.add_paragraph(
    "Однако вместо прямого импорта mvpose (Cython, CUDA, жёсткая конфигурация) мы "
    "ПЕРЕРЕАЛИЗУЕМ ядро алгоритмов, опираясь на mvpose-подход:"
)

items = [
    "Эпиполярное расстояние: cv2.computeCorrespondEpilines() + среднее расстояние "
    "точки до линии (по mvpose geometry.py:29-61)",
    "Матрица сходства: z-score нормализация + сигмоида "
    "(по mvpose geometry.py:93-96)",
    "Венгерское назначение через scipy.optimize.linear_sum_assignment "
    "(вместо SVT для простоты — SVT избыточен для ≤6 камер, ≤5 людей)",
    "Транзитивное замыкание для многокамерного консенсуса "
    "(по mvpose pictorial.pyx:201-234)",
]

for item in items:
    doc.add_paragraph(item, style='List Bullet')

doc.add_paragraph(
    "Этот подход исключает зависимость от обученных весов, ONNX-моделей и CUDA, "
    "сохраняя математическую строгость проверенного алгоритма mvpose."
)

doc.save(path)
print(f"Section 13 appended to {path}")
