import os
import PIL.Image
import cv2
import einops
import numpy as np
import torch
import random
import math
import PIL
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import argparse
import datetime
import string
import time
from dataset.opencv_transforms.functional import to_tensor, center_crop

from pytorch_lightning import seed_everything
from vtdm.model import create_model
from vtdm.util import tensor2vid

models = {}

seed = random.randint(0, 65535)
seed_everything(seed)

import time
stamp = int(time.time())

parser = argparse.ArgumentParser()
parser.add_argument('--denoise_config', type=str, default="configs/multiview/inference_low.yaml")
parser.add_argument('--denoise_checkpoint', type=str, default="/data/gyj/hi3d_256/outputs/logs/train-v01/stage1_arcface/checkpoints/trainstep_checkpoints/epoch=000071-step=000000999.ckpt/global_step1000/mp_rank_00_model_states.pt")
parser.add_argument('--image_path', type=str, default="examples")
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument('--elevation', type=int, default=0)
params = parser.parse_args()

denoise_config = params.denoise_config
denoise_checkpoint = params.denoise_checkpoint

denoising_model = create_model(denoise_config).cpu()
denoising_model.init_from_ckpt(denoise_checkpoint)
denoising_model = denoising_model.cuda().half()
                                                 
models['denoising_model'] = denoising_model

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "./ckpts/ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")


transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def preprocess(image: Image.Image) -> Image.Image:
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image
     
def denoising(frames, aes, mv, elevation):
    with torch.no_grad():        
        C, T, H, W = frames.shape
        clip_size = models['denoising_model'].num_samples
        assert T == clip_size
        
        batch = {'video': frames.unsqueeze(0)}
        batch['elevation'] = torch.Tensor([elevation]).to(torch.int64).to(frames.device)
        batch['fps_id'] = torch.Tensor([7]).to(torch.int64).to(frames.device)
        batch['motion_bucket_id'] = torch.Tensor([127]).to(torch.int64).to(frames.device)
        batch = models['denoising_model'].add_custom_cond(batch, infer=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            c, uc = models['denoising_model'].conditioner.get_unconditional_conditioning(
                batch,
                force_uc_zero_embeddings=['cond_frames', 'cond_frames_without_noise']
            )

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            2, clip_size
        ).to(models['denoising_model'].device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
        def denoiser(input, sigma, c):
            return models['denoising_model'].denoiser(
                models['denoising_model'].model, input, sigma, c, **additional_model_inputs
            )
            
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            randn = torch.randn([T, 4, H // 8, W // 8], device=models['denoising_model'].device)
            samples = models['denoising_model'].sampler(denoiser, randn, cond=c, uc=uc)
        
        samples = models['denoising_model'].decode_first_stage(samples.half())
        
        samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=clip_size)
        
    return tensor2vid(samples)


def video_pipeline(frames, key, args):
    # seed = args['seed']
    num_iter = args['num_iter']
    
    out_list = []
    for it in range(num_iter):
        
        with torch.no_grad():
            results = denoising(frames, args['aes'], args['mv'], args['elevation'])
               
        if len(out_list) == 0:
            out_list = out_list + results
        else:
            out_list = out_list + results[1:]
        img = out_list[-1]
        img = to_tensor(img)
        img = (img - 0.5) * 2.0
        frames[:, 0] = img
    
    output_videos_dir = args["output_dir"]
    output_videos_dir = os.path.join(output_videos_dir, 'low_stage')
    os.makedirs(output_videos_dir, exist_ok=True)
    for i in range(16):
        frame_path = os.path.join(output_videos_dir, f"{(i+16):04d}.png")
        img = cv2.cvtColor(out_list[i],cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, img)
            
def process(args, key='image'):
    image_path = args['image_path']

    img = cv2.imread(image_path)                                                 # 2048,2028,3
    frame_list = [img] * args['clip_size']

    h, w = frame_list[0].shape[0:2]
    rate = max(args['input_resolution'][0] * 1.0 / h, args['input_resolution'][1] * 1.0 / w)
    frame_list = [cv2.resize(f, [math.ceil(w * rate), math.ceil(h * rate)]) for f in frame_list]
    frame_list = [center_crop(f, [args['input_resolution'][0], args['input_resolution'][1]]) for f in frame_list]
    frame_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_list]
    
    frame_list = [to_tensor(f) for f in frame_list]
    frame_list = [(f - 0.5) * 2.0 for f in frame_list]
    frames = torch.stack(frame_list, 1)
    frames = frames.cuda()

    models['denoising_model'].num_samples = args['clip_size']
    models['denoising_model'].image_size = args['input_resolution']
    
    video_pipeline(frames, key, args)

if os.path.isdir(params.image_path):
    image_path_list = [os.path.join(params.image_path, sub) for sub in os.listdir(params.image_path)]
else:
    image_path_list = [params.image_path]

for image_path in image_path_list:

    basename = os.path.basename(image_path).split(".")[0]
    out_path = os.path.join(params.output_dir, basename)
    print(f"image path: {image_path} out path: {out_path}")
    os.makedirs(out_path, exist_ok=True)
    temp_image_dir = os.path.join(params.output_dir, basename,"temp_image")
    os.makedirs(temp_image_dir, exist_ok=True)
    image = PIL.Image.open(image_path)

    if image.mode == "RGB":
        image = preprocess(image)

    white_image_path = os.path.join(temp_image_dir, "white.png")
    white_image = Image.new("RGB", image.size, "WHITE")
    white_image.paste(image, mask=image.split()[3])
    white_image.save(white_image_path)

    # 3. first step , generate first image, and save in "args.output_dir/first_step/first.mp4"
    infer_config = {
            "image_path": white_image_path,
            "clip_size": 16,
            "input_resolution": [
                256,
                256
            ],
            "num_iter": 1,
            "seed": -1,
            "aes": 6.0,
            "mv": [
                0.0,
                0.0,
                0.0,
                10.0
            ],
            "elevation": params.elevation,
            "output_dir": out_path
    }

    process(infer_config)