import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
from ultralytics import YOLO
import numpy as np
import os
import sys
from collections import Counter
import time

client = OpenAI(
    api_key="sk-or-vv-7952300267ec5efa67eab60c3d6504cfa712007953e185d3440cc09767a6e503",
    base_url="https://api.vsegpt.ru/v1"
)
FRAME_RATE = 1
MAX_FRAMES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt")
tracked_objects = {}


def extract_frames(video_path, frame_rate=FRAME_RATE, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return []

    # с GitHub
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Видео: {duration:.1f}с, {fps:.1f} FPS, {total_frames} кадров")

    step = int(max(1, fps / frame_rate))
    frames = []
    timestamps = []
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            timestamp = count / fps
            frames.append(frame)
            timestamps.append(timestamp)
        count += 1
    cap.release()
    return frames, timestamps


def init_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    return processor, model


def describe_frame_blip(frame, processor, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=50)  # с GitHub
    caption = processor.decode(out_ids[0], skip_special_tokens=True)
    return caption


def get_dominant_color(crop):
    if crop.size == 0:
        return "RGB(неизвестно)"

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (50, 50))
    pixels = crop.reshape(-1, 3)
    most_common = Counter([tuple(p) for p in pixels]).most_common(1)[0][0]

    r, g, b = most_common
    return f"RGB({r},{g},{b})"


def detect_objects_and_colors_enhanced(frame, timestamp, frame_idx):
    """Улучшенная детекция с трекингом как в GitHub проекте"""
    # Используем track вместо обычного вызова для связи объектов между кадрами
    results = yolo_model.track(frame, persist=True, conf=0.5, verbose=False)

    if not results or results[0].boxes is None:
        return []

    descriptions = []
    result = results[0]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].int().tolist()
        x1, y1, x2, y2 = xyxy

        # Получаем ID трекинга
        track_id = int(box.id[0]) if box.id is not None else None

        obj_crop = frame[y1:y2, x1:x2]
        color = get_dominant_color(obj_crop)

        # размер
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame.shape[0] * frame.shape[1]
        relative_size = area / frame_area

        if relative_size > 0.1:
            size_desc = "большой"
        elif relative_size > 0.05:
            size_desc = "средний"
        else:
            size_desc = "маленький"

        desc = f"{size_desc} {color} {label}"
        if confidence > 0.8:
            desc += " (четко видимый)"
        if track_id is not None:
            desc += f" [ID:{track_id}]"

        descriptions.append(desc)

        if track_id is not None:
            if track_id not in tracked_objects:
                tracked_objects[track_id] = []
            tracked_objects[track_id].append({
                'timestamp': timestamp,
                'frame_idx': frame_idx,
                'label': label,
                'position': [(x1 + x2) // 2, (y1 + y2) // 2],
                'confidence': confidence
            })

    return descriptions


def analyze_object_movements():
    """Анализ движения объектов между кадрами"""
    movements = []

    for track_id, trajectory in tracked_objects.items():
        if len(trajectory) < 2:
            continue

        # траектория
        first_pos = trajectory[0]['position']
        last_pos = trajectory[-1]['position']
        first_time = trajectory[0]['timestamp']
        last_time = trajectory[-1]['timestamp']

        # Вычисляем смещение
        dx = last_pos[0] - first_pos[0]
        dy = last_pos[1] - first_pos[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        if distance > 50:
            direction = ""
            if abs(dx) > abs(dy):
                direction = "вправо" if dx > 0 else "влево"
            else:
                direction = "вниз" if dy > 0 else "вверх"

            label = trajectory[0]['label']
            movements.append(f"{label} движется {direction} с {first_time:.1f}с по {last_time:.1f}с")

    return movements


def detect_scene_changes(prev_descriptions, curr_descriptions):
    """Определение изменений в сцене"""
    prev_objects = set(desc.split()[2] for desc in prev_descriptions if len(desc.split()) > 2)
    curr_objects = set(desc.split()[2] for desc in curr_descriptions if len(desc.split()) > 2)

    changes = []

    # Новые объекты
    new_objects = curr_objects - prev_objects
    for obj in new_objects:
        changes.append(f"появился {obj}")

    # Исчезнувшие объекты
    disappeared = prev_objects - curr_objects
    for obj in disappeared:
        changes.append(f"исчез {obj}")

    return changes


def ask_gpt_enhanced(captions_with_time, movements, user_question):
    """Улучшенный запрос к GPT"""
    timed_descriptions = []
    for i, (caption, timestamp) in enumerate(captions_with_time):
        timed_descriptions.append(f"[{timestamp:.1f}с] {caption}")

    joined = "\n".join(f"- {desc}" for desc in timed_descriptions)

    # Добавляем информацию о движениях
    movements_text = ""
    if movements:
        movements_text = "\n\nДВИЖЕНИЯ ОБЪЕКТОВ:\n" + "\n".join(f"- {mov}" for mov in movements)

    # Статистика объектов
    all_objects = []
    for caption, _ in captions_with_time:
        if "Objects:" in caption:
            objects_part = caption.split("Objects:")[1].split(";")[0]
            all_objects.extend([obj.strip().split()[-1] for obj in objects_part.split(",") if obj.strip()])

    object_stats = Counter(all_objects)
    stats_text = ""
    if object_stats:
        most_common = object_stats.most_common(3)
        stats_text = f"\n\nЧАСТО ВСТРЕЧАЮЩИЕСЯ ОБЪЕКТЫ: {', '.join([f'{obj}({count})' for obj, count in most_common])}"

    prompt = (
        "Ниже приведены описания кадров видео с временными метками и детекцией объектов:\n"
        f"{joined}"
        f"{movements_text}"
        f"{stats_text}\n\n"
        f"Вопрос: {user_question}\n"
        "Ответь подробно, используя временные метки и информацию о движении объектов, переводи формат RGB() в "
        "реальный цвет при ответе, например (RGB(22,15,25)) писать не надо"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "Ты — эксперт по анализу видео. Отвечай подробно, используя временные метки и описания движений."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при запросе к OpenAI: {e}"


def main():
    video_path = "./concert.mp4"
    if not os.path.exists(video_path):
        print(f"Видео файл не найден: {video_path}")
        return

    user_question = input("Введите ваш вопрос о видео: ").strip()
    if not user_question:
        print("Вопрос не указан.")
        return

    print("Извлечение кадров из видео...")
    frames, timestamps = extract_frames(video_path)
    if not frames:
        print("Не удалось извлечь кадры.")
        return
    print(f"Извлечено {len(frames)} кадров.")

    processor, model = init_blip()

    final_captions = []
    prev_object_descriptions = []
    all_changes = []

    print("Анализ кадров...")
    for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        try:
            start_time = time.time()

            caption_blip = describe_frame_blip(frame, processor, model)

            object_descs = detect_objects_and_colors_enhanced(frame, timestamp, idx)

            if prev_object_descriptions:
                changes = detect_scene_changes(prev_object_descriptions, object_descs)
                if changes:
                    all_changes.extend([f"[{timestamp:.1f}с] {change}" for change in changes])

            objects_text = ', '.join(object_descs) if object_descs else 'нет объектов'
            merged = f"BLIP: {caption_blip}; Objects: {objects_text}"

            final_captions.append((merged, timestamp))
            prev_object_descriptions = object_descs

            process_time = time.time() - start_time
            print(f"Кадр {idx + 1} ({timestamp:.1f}с): {len(object_descs)} объектов за {process_time:.2f}с")

        except Exception as e:
            print(f"Ошибка в кадре {idx + 1}: {e}")
            final_captions.append((f"[ошибка анализа кадра {idx + 1}]", timestamp))

    movements = analyze_object_movements()

    if movements:
        print("\nОбнаруженные движения:")
        for movement in movements:
            print(f"- {movement}")

    if all_changes:
        print("\nИзменения в сцене:")
        for change in all_changes:
            print(f"- {change}")

    answer = ask_gpt_enhanced(final_captions, movements + all_changes, user_question)
    print("\n=== Ответ модели ===")
    print(answer)


if __name__ == "__main__":
    main()