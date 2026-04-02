import os
import cv2
import einops
import torch
import random
import math
import PIL
import argparse
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from dataset.opencv_transforms.functional import to_tensor, center_crop
from pytorch_lightning import seed_everything
from vtdm.model import create_model
from vtdm.util import tensor2vid

def get_args():
    parser = argparse.ArgumentParser(description="Full Multi-view Inference Pipeline")
    parser.add_argument('--low_config', type=str, default="configs/multiview/inference_low.yaml")
    parser.add_argument('--low_ckpt', type=str, default="ckpts/low_stage.pt")
    parser.add_argument('--high_config', type=str, default="configs/multiview/inference-high.yaml")
    parser.add_argument('--high_ckpt', type=str, default="ckpts/high_stage.pt")
    parser.add_argument('--input_path', type=str, default="examples", help="Input image or directory")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument('--elevation', type=int, default=0)
    return parser.parse_args()

class FullPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading BiRefNet...")
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "./ckpts/ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(self.device)
        self.rmbg_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print("Loading Low-Res Model...")
        self.model_low = create_model(args.low_config).cpu()
        self.model_low.init_from_ckpt(args.low_ckpt)
        self.model_low = self.model_low.to(self.device).half()
        
        print("Loading High-Res Model...")
        self.model_high = create_model(args.high_config).cpu()
        self.model_high.init_from_ckpt(args.high_ckpt)
        self.model_high = self.model_high.to(self.device).half()

    def remove_background(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGBA":
            input_tensor = self.rmbg_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.birefnet(input_tensor)[-1].sigmoid().cpu()
            mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)
            image.putalpha(mask)
        
        white_bg = Image.new("RGB", image.size, "WHITE")
        white_bg.paste(image, mask=image.split()[3])
        return white_bg

    def prepare_input_frames(self, img_cv2, resolution):
        h, w = img_cv2.shape[0:2]
        rate = max(resolution[0] / h, resolution[1] / w)
        img_res = cv2.resize(img_cv2, [math.ceil(w * rate), math.ceil(h * rate)])
        img_res = center_crop(img_res, [resolution[0], resolution[1]])
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        
        tensor_frame = (to_tensor(img_res) - 0.5) * 2.0
        frames = torch.stack([tensor_frame] * 16, 1).to(self.device)
        return frames

    def run_denoising(self, model, frames, elevation, stage1_frames_embedding=None):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            C, T, H, W = frames.shape
            batch = {
                'video': frames.unsqueeze(0),
                'elevation': torch.Tensor([elevation]).to(torch.int64).to(self.device),
                'fps_id': torch.Tensor([7]).to(torch.int64).to(self.device),
                'motion_bucket_id': torch.Tensor([127]).to(torch.int64).to(self.device),
                'num_video_frames': T
            }
            
            if stage1_frames_embedding is not None:
                batch['video_stage1_embedding'] = stage1_frames_embedding 

            batch = model.add_custom_cond(batch, infer=True)
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, force_uc_zero_embeddings=['cond_frames', 'cond_frames_without_noise']
            )

            extra_inputs = {
                "image_only_indicator": torch.zeros(2, T).to(self.device),
                "num_video_frames": T
            }

            def denoiser_wrapper(input, sigma, c):
                return model.denoiser(model.model, input, sigma, c, **extra_inputs)

            randn = torch.randn([T, 4, H // 8, W // 8], device=self.device)
            if stage1_frames_embedding is not None:
                randn = 0.5 * randn + 0.5 * batch['frames_stage1_embedding']

            samples = model.sampler(denoiser_wrapper, randn, cond=c, uc=uc)
            samples = model.decode_first_stage(samples.half())
            samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=T)
            return tensor2vid(samples)

    def process_single_image(self, image_path):
        basename = os.path.basename(image_path).split(".")[0]
        subj_dir = os.path.join(self.args.output_dir, basename)
        os.makedirs(subj_dir, exist_ok=True)

        raw_img = Image.open(image_path)
        white_img_pil = self.remove_background(raw_img)
        white_img_cv2 = cv2.cvtColor(np.array(white_img_pil), cv2.COLOR_RGB2BGR)

        print(f"[{basename}] Running Stage 1...")
        frames_low = self.prepare_input_frames(white_img_cv2, [256, 256])
        self.model_low.num_samples = 16
        res_low_list = self.run_denoising(self.model_low, frames_low, self.args.elevation)

        low_stage_dir = os.path.join(subj_dir, "low_stage")
        os.makedirs(low_stage_dir, exist_ok=True)
        stage1_tensors = []
        for i, img_np in enumerate(res_low_list):
            cv2.imwrite(os.path.join(low_stage_dir, f"{(i+16):04d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            t_img = (to_tensor(cv2.resize(img_np, (512, 512))) - 0.5) * 2.0
            stage1_tensors.append(t_img)
        
        frames_256_for_high = torch.stack(stage1_tensors, 1).to(self.device)

        print(f"[{basename}] Running Stage 2...")
        frames_high = self.prepare_input_frames(white_img_cv2, [512, 512])
        self.model_high.num_samples = 16
        
        with torch.no_grad():
            res_high_list = self.denoising_high_wrapper(frames_high, frames_256_for_high)

        high_stage_dir = os.path.join(subj_dir, "high_stage")
        os.makedirs(high_stage_dir, exist_ok=True)
        for i, img_np in enumerate(res_high_list):
            cv2.imwrite(os.path.join(high_stage_dir, f"{(i+16):04d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    def denoising_high_wrapper(self, frames, frames_256):
        model = self.model_high
        T = 16
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            batch = {
                'video': frames.unsqueeze(0),
                'video_stage1': frames_256.unsqueeze(0),
                'elevation': torch.Tensor([self.args.elevation]).to(torch.int64).to(self.device),
                'fps_id': torch.Tensor([7]).to(torch.int64).to(self.device),
                'motion_bucket_id': torch.Tensor([127]).to(torch.int64).to(self.device),
                'num_video_frames': T
            }
            batch = model.add_custom_cond(batch, infer=True)
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, force_uc_zero_embeddings=['cond_frames', 'cond_frames_without_noise']
            )
            
            extra_inputs = {"image_only_indicator": torch.zeros(2, T).to(self.device), "num_video_frames": T}
            def denoiser(input, sigma, c):
                return model.denoiser(model.model, input, sigma, c, **extra_inputs)

            randn = torch.randn([T, 4, 512 // 8, 512 // 8], device=self.device)
            randn = 0.5 * randn + 0.5 * batch['frames_stage1_embedding']
            
            samples = model.sampler(denoiser, randn, cond=c, uc=uc)
            samples = model.decode_first_stage(samples.half())
            samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=T)
            return tensor2vid(samples)

if __name__ == "__main__":
    args = get_args()
    seed_everything(random.randint(0, 65535))
    
    pipeline = FullPipeline(args)
    
    if os.path.isdir(args.input_path):
        img_list = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_list = [args.input_path]

    for img_path in img_list:
        print(f"Processing: {img_path}")
        pipeline.process_single_image(img_path)