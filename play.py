import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

from pictures import picturin
#torch.cuda.set_device(0)  # set the GPU device




# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "xyn-ai/anything-v4.0"
# model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)




from masactrl.masactrl import MutualSelfAttentionControl

'''
prompts = [
    "1boy, casual, outdoors, ",  # source prompt
    "1boy, casual, outdoors, sitting on a bench", # mid 1
    "1boy, casual, outdoors, sitting on a bench, surrounded by flowers", # mid 2
    "1boy, casual, outdoors, sitting on a bench, surrounded by flowers, in a rainy day",# target prompt
]
prompts_data.append(prompts)
'''

prompts_data=[]

prompts = [
    "1man",  # source prompt
    "1man, walking down streets", # mid 1
    "1man, walking down streets, with his dog", # mid 2
    "1man, walking down streets, with his dog, in winter",# target prompt
]
prompts_data.append(prompts)
prompts = [
    "in a large room",  # source prompt
    "in a large room, many people celebrating", # mid 1
    "in a large room, many people celebrating, a girl at the central, ", # mid 2
    "in a large room, many people celebrating, a girl at the central, holding a birthday cake",# target prompt
]
prompts_data.append(prompts)
prompts = [
    "on the summit",  # source prompt
    "on the summit, covered with snow", # mid 1
    "on the summit, covered with snow, 1girl ", # mid 2
    "on the summit, covered with snow, 1girl, skiing downhill",# target prompt
]
prompts_data.append(prompts)
prompts = [
    "1man in plaid shirt",  # source prompt
    "1man in plaid shirt, 1girl in long dress", # mid 1
    "1man in plaid shirt, 1girl in long dress, 1old man with long white bear", # mid 2
    "1man in plaid shirt, 1girl in long dress, 1old man with long white bear , holding a camera",# target prompt
]
prompts_data.append(prompts)
prompts = [
    "in a zoo",  # source prompt
    "in a zoo, 1tiger", # mid 1
    "in a zoo, 1tiger, 1lion", # mid 2
    "in a zoo, 1tiger, 1lion, 1bear",# target prompt
]


image_reader=picturin("./in")
image=image_reader()
#image=None
#print(image)


prompts_data.append(prompts)
count=0
for prompts in prompts_data:
    
    seed = 42+count
    seed_everything(seed)
    
    count+=1
    out_dir = "./workdir/masactrl_exp/"
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count}, {count}")
    os.makedirs(out_dir, exist_ok=True)

    start_code = torch.randn([1, 4, 64, 64], device=device)
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    STEP = 24
    LAYPER = 8
    total_steps=50

    # hijack the attention module
    editor = MutualSelfAttentionControl(STEP, LAYPER, total_steps=total_steps)
    regiter_attention_editor_diffusers(model, editor)

    # inference the synthesized image
    image_masactrl = model(image, prompts, latents=start_code, guidance_scale=7.5,num_inference_steps=total_steps)

    # save the synthesized image
    #out_image = torch.cat([image_ori, image_masactrl], dim=0)
    out_image = image_masactrl
    save_image(out_image, os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}.png"))
    # save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
    # save_image(out_image[1], os.path.join(out_dir, f"without_step{STEP}_layer{LAYPER}.png"))W
    # save_image(out_image[2], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYPER}.png"))

    print("Syntheiszed images are saved in", out_dir)


