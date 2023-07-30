from diffusers import DiffusionPipeline
import torch
import torchvision
import gradio as gr


device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to(device)


def generate_image(prompt):
  image = pipe(prompt=prompt).images[0]
  return(image)


title = "Image generation"
description = "This System allows you to enter text & generate image based on it."

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():

        with gr.Group():
            prompt = gr.Textbox(value="A majestic lion jumping from a big stone at night", label="What do you want to generate?")
            btn = gr.Button(value="Submit")
            btn.style(full_width=True)

        with gr.Group():
            image = gr.Image(label="answer")

        btn.click(generate_image, inputs=[prompt], outputs=[image])

demo.queue()

demo.launch(share=True)
