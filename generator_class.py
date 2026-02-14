import torch
from diffusers import DiffusionPipeline
import time
from PIL import Image
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def generate_image_fast(
        prompt="a beautiful tree",
        output_path="tree.png",
):
    print(f"Используется устройство: CPU")
    print(f"PyTorch версия: {torch.__version__}")

    start_time = time.time()

    # Используем Tiny AutoEncoder модель
    model_id = "OFA-Sys/small-stable-diffusion-v0"

    print("Загрузка модели...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float32,  # Явно указываем float32 для CPU
    )

    # Перемещаем на CPU
    pipe = pipe.to("cpu")

    print("Генерация изображения...")
    # Уменьшаем параметры для скорости
    image = pipe(
        prompt,
        num_inference_steps=10,  # Мало шагов
        height=256,
        width=256,
        guidance_scale=6.0,
    ).images[0]

    # Увеличиваем размер
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image.save(output_path)

    print(f"✅ Готово за {time.time() - start_time:.1f} сек")
    print(f"📁 Изображение: {output_path}")
