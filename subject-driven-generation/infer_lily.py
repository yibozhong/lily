from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch
import os
from peft import get_peft_model, LilyConfig
model = "stabilityai/stable-diffusion-xl-base-1.0"

def load_lily(model, input_dir, lily_config):
    saved_parameters = torch.load(os.path.join(input_dir, 'lily_layers.pt'))
    model = get_peft_model(model, lily_config)
    param_diff = 0
    for n, p in model.named_parameters():
        if not 'lily_' in n:
            continue
        n_ = n.replace('base_model.model.', '')
        if n_ in saved_parameters:
            p.data.copy_(saved_parameters[n_].data)
        else:
            print(f"Warning: Parameter {n_} not found in model state dict.")
            param_diff += p.numel()
    print(f"{param_diff} differet parameters")
    param = 0
    for n, p in model.named_parameters():
        if 'lily_' in n:
            param += p.numel()
    print(f"param is {param}")
    return model

subject="robot_toy"
templates = [
    "A picture of a {} in the jungle",
    "A picture of a {} in the snow",
    "A picture of a {} on the beach",
    "A picture of a {} on a cobblestone street",
    "A picture of a {} on top of pink fabric",
    "A picture of a {} on top of a wooden floor",
    "A picture of a {} with a city in the background",
    "A picture of a {} with a mountain in the background",
    "A picture of a {} with a blue house in the background",
    "A picture of a {} on top of a purple rug in a forest",
    "A picture of a {} with a wheat field in the background",
    "A picture of a {} with a tree and autumn leaves in the background",
    "A picture of a {} with the Eiffel Tower in the background",
    "A picture of a {} floating on top of water",
    "A picture of a {} floating in an ocean of milk",
    "A picture of a {} on top of green grass with sunflowers around it",
    "A picture of a {} on top of a mirror",
    "A picture of a {} on top of the sidewalk in a crowded street",
    "A picture of a {} on top of a dirt road",
    "A picture of a {} on top of a white rug",
    "A picture of a red {}",
    "A picture of a purple {}",
    "A picture of a shiny {}",
    "A picture of a wet {}",
    "A picture of a cube shaped {}",
]


prompts = [template.format(subject) for template in templates]


for method in ["lily"]:
    if method == "lily":
        path = "./lily-trained-xl-{}".format(subject.replace(" ", "_"))

    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    unet_lily_config = LilyConfig(
        r=4,
        ne_1=4,
        ne_2=4,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet = load_lily(pipe.unet, path, unet_lily_config)
    pipe = pipe.to("cuda")
    for prompt in prompts:
        print(prompt)
        image = pipe(prompt, num_inference_steps=50).images[0]

        output_p_dir="output_images/lily/{}/{}".format(subject, prompt.replace(" ", "_"))

        os.makedirs(output_p_dir, exist_ok=True)

        image.save("{}/{}.jpg".format(output_p_dir, method))