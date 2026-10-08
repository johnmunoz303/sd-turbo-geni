import torch  # Ejecuta operaciones matemáticas del modelo de IA
from diffusers import AutoPipelineForText2Image
# Diffusers es una librería especializada en modelos generativos

print("Cargando el modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32,  # Precisión numérica de los pesos
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")

negative_prompt = (
    "worst quality, low quality, bad quality, low resolution, "
    "blurry, distorted, deformed body, bad anatomy, deformed limbs, "
    "badly drawn hands, extra fingers, missing fingers, "
    "text, watermark, signature, logo, compression artifacts"
)

print("Generando imagen...")

imagen = modelo(
    prompt=prompt,                     # Instrucciones para crear la imagen
    negative_prompt=negative_prompt,
    num_inference_steps=10,             # Número de pasos de generación
    guidance_scale=2.0,                  # Intensidad de la guía del prompt
    width=1024,
    height=1024,
).images[0]

imagen.save("imagen.png")

print("Imagen guardada como imagen.png")
