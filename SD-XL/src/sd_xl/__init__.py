import torch 
from diffusers import AutoPipelineForText2Image

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear:")
negative_prompt= "worst quality, low quality:1.4, bad anatomy, deformed limbs, badly drawn hands, extra fingers"
 
print("Generando imagen ...")

imagen= modelo(
       prompt=prompt,
       # negative _prompt=negative_prompt,
       num_inference_steps=25, 
       guidance_scale=7.0,
       height=1024,
       width=1024
).images[0]

imagen.save("imagen.png")
 
print("Imagen guardada como imagen.png")
