import os
import cv2
import time
import yaml
import torch
import random
import math
import argparse
import numpy as np
import einops
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import json

# --- Imports from MV Generation ---
from dataset.opencv_transforms.functional import to_tensor, center_crop
from pytorch_lightning import seed_everything
from vtdm.model import create_model
from vtdm.util import tensor2vid

# --- Imports from Reconstruction ---
from core.options import Options
from core.models import LGM
from safetensors.torch import load_file

# -----------------------------------------------------------------------------
# Unified Argument Parser
# -----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Unified Multi-view Generation & 3D Reconstruction Pipeline")
    
    # Common Args
    parser.add_argument('--input_path', type=str, default="examples", help="Input image or directory")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument('--seed', type=int, default=None)

    # MV Generation Args
    parser.add_argument('--low_config', type=str, default="configs/multiview/inference_low.yaml")
    parser.add_argument('--low_ckpt', type=str, default="ckpts/low_stage.pt")
    parser.add_argument('--high_config', type=str, default="configs/multiview/inference-high.yaml")
    parser.add_argument('--high_ckpt', type=str, default="ckpts/high_stage.pt")
    parser.add_argument('--elevation', type=int, default=0)

    # Reconstruction Args
    parser.add_argument('--recon_config', type=str, default="configs/recon/inference.yaml")
    parser.add_argument('--recon_resume', type=str, default="ckpts/recon.safetensors", help="Path to LGM checkpoint")
    parser.add_argument('--lr', type=float, default=1e-6, help="Learning rate for reconstruction")
    parser.add_argument('--num_epochs', type=int, default=500, help="Number of epochs for reconstruction fine-tuning")
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Unified Pipeline Class
# -----------------------------------------------------------------------------
class UnifiedPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing Pipeline on {self.device}...")

        # 1. Load BiRefNet (Shared Resource)
        print("[INFO] Loading BiRefNet (Shared)...")
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "./ckpts/ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(self.device)
        self.birefnet.eval()

        self.rmbg_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 2. Load MV Generation Models
        print("[INFO] Loading MV Generation Models (Low & High Res)...")
        self.model_low = create_model(args.low_config).cpu()
        self.model_low.init_from_ckpt(args.low_ckpt)
        self.model_low = self.model_low.to(self.device).half()
        
        self.model_high = create_model(args.high_config).cpu()
        self.model_high.init_from_ckpt(args.high_ckpt)
        self.model_high = self.model_high.to(self.device).half()

        # 3. Prepare Reconstruction Options (Configuration)
        print("[INFO] Loading Reconstruction Configuration...")
        with open(args.recon_config, 'r') as f:
            recon_config_dict = yaml.safe_load(f)
        self.recon_opts = Options(**recon_config_dict)
        self.recon_opts.resume = args.recon_resume
        self.recon_opts.lr = args.lr
        self.recon_opts.num_epochs = args.num_epochs
        # Workspace will be set dynamically per subject

    # =========================================================================
    # Part 1: Helper Functions (Image Processing)
    # =========================================================================
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Removes background using BiRefNet."""
        # Check if we need to remove background (if not RGBA) or just for cleanup
        if image.mode != "RGBA":
             image = image.convert("RGB")
        
        image_size = image.size
        input_tensor = self.rmbg_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet(input_tensor)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(image_size)
        
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask)
        
        # For MV Gen: we often need a white background composite
        white_bg = Image.new("RGB", image_size, "WHITE")
        white_bg.paste(image_rgba, mask=mask)
        
        return white_bg, image_rgba # Return both white-bg (for MV) and transparent (for Recon)

    def prepare_input_frames(self, img_cv2, resolution):
        h, w = img_cv2.shape[0:2]
        rate = max(resolution[0] / h, resolution[1] / w)
        img_res = cv2.resize(img_cv2, [math.ceil(w * rate), math.ceil(h * rate)])
        img_res = center_crop(img_res, [resolution[0], resolution[1]])
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        
        tensor_frame = (to_tensor(img_res) - 0.5) * 2.0
        frames = torch.stack([tensor_frame] * 16, 1).to(self.device)
        return frames

    # =========================================================================
    # Part 2: Multi-View Generation Logic
    # =========================================================================

    def run_denoising(self, model, frames, elevation, stage1_frames_embedding=None, resolution=256):
        T = 16
        H, W = (resolution, resolution)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            batch = {
                'video': frames.unsqueeze(0),
                'elevation': torch.Tensor([elevation]).to(torch.int64).to(self.device),
                'fps_id': torch.Tensor([7]).to(torch.int64).to(self.device),
                'motion_bucket_id': torch.Tensor([127]).to(torch.int64).to(self.device),
                'num_video_frames': T
            }
            
            # Add stage1 conditioning for stage 2
            if stage1_frames_embedding is not None:
                # Assuming the model wrapper handles adding 'video_stage1' if needed or embeddings
                # Note: The original code passed 'video_stage1' in batch for stage 2 wrapper
                # We will adapt logic below in specific wrappers or keep this generic
                pass 

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
                 # In stage 2, we mix noise with stage 1 embedding
                randn = 0.5 * randn + 0.5 * batch['frames_stage1_embedding']

            samples = model.sampler(denoiser_wrapper, randn, cond=c, uc=uc)
            samples = model.decode_first_stage(samples.half())
            samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=T)
            return tensor2vid(samples)
            
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

    def run_mv_generation(self, image_path):
            basename = os.path.basename(image_path).split(".")[0]
            subj_dir = os.path.join(self.args.output_dir, basename)
            os.makedirs(subj_dir, exist_ok=True)
            
            print(f"[{basename}] Processing Image: {image_path}")
            
            # 1. Preprocess
            raw_img = Image.open(image_path)
            white_img_pil, _ = self.remove_background(raw_img)
            white_img_cv2 = cv2.cvtColor(np.array(white_img_pil), cv2.COLOR_RGB2BGR)

            # 2. Stage 1 (Low Res)
            print(f"[{basename}] Running MV Stage 1 (Low Res)...")

            self.model_low = self.model_low.to(self.device) 
            frames_low = self.prepare_input_frames(white_img_cv2, [256, 256])
            self.model_low.num_samples = 16
            res_low_list = self.run_denoising(self.model_low, frames_low, self.args.elevation, resolution=256)

            print(f"[{basename}] Offloading Low-Res model to CPU...")
            self.model_low = self.model_low.cpu()
            torch.cuda.empty_cache() 

            low_stage_dir = os.path.join(subj_dir, "low_stage")
            os.makedirs(low_stage_dir, exist_ok=True)
            stage1_tensors = []
            for i, img_np in enumerate(res_low_list):
                cv2.imwrite(os.path.join(low_stage_dir, f"{(i+16):04d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                t_img = (to_tensor(cv2.resize(img_np, (512, 512))) - 0.5) * 2.0
                stage1_tensors.append(t_img)
            
            frames_256_for_high = torch.stack(stage1_tensors, 1).to(self.device)

            # 3. Stage 2 (High Res)
            print(f"[{basename}] Running MV Stage 2 (High Res)...")

            self.model_high = self.model_high.to(self.device)
            frames_high = self.prepare_input_frames(white_img_cv2, [512, 512])
            self.model_high.num_samples = 16
            
            res_high_list = self.denoising_high_wrapper(frames_high, frames_256_for_high)

            print(f"[{basename}] Offloading High-Res model to CPU...")
            self.model_high = self.model_high.cpu()
        
            
            torch.cuda.empty_cache()

            high_stage_dir = os.path.join(subj_dir, "high_stage")
            os.makedirs(high_stage_dir, exist_ok=True)
            
            for i, img_np in enumerate(res_high_list):
                save_path = os.path.join(high_stage_dir, f"{(i+16):04d}.png")
                cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            print(f"[{basename}] MV Generation Complete. Output: {subj_dir}")
            return subj_dir

    # =========================================================================
    # Part 3: Reconstruction Logic (LGM)
    # =========================================================================

    def preprocess_for_recon(self, image_path):
        """Ensure images in high_stage have alpha channel for LGM."""
        img = Image.open(image_path)
        if img.mode != "RGBA":
            # Re-use the shared birefnet to create alpha mask
            _, img_rgba = self.remove_background(img)
            img_rgba.save(image_path)

    def run_reconstruction(self, subject_dir):
        print(f"[RECON] Starting Reconstruction for: {subject_dir}")
        
        # 1. Update Options for this subject
        opt = self.recon_opts
        opt.subject = subject_dir # LGM dataset loader uses this to find 'high_stage'
        opt.workspace = subject_dir # Save results in the same subject folder
        
        # 2. Preprocess MV images (Add Alpha Channel if missing)
        high_stage_path = os.path.join(subject_dir, "high_stage")
        if not os.path.exists(high_stage_path):
            print(f"[ERROR] high_stage directory not found in {subject_dir}. Skipping.")
            return

        print(f"[RECON] Preprocessing frames (masking) in {high_stage_path}...")
        for image_name in os.listdir(high_stage_path):
            if image_name.endswith(('.png', '.jpg')):
                self.preprocess_for_recon(os.path.join(high_stage_path, image_name))

        # 3. Initialize Model
        print(f"[RECON] Loading LGM Model...")
        model = LGM(opt)
        
        # Load Checkpoint
        if opt.resume is not None:
            if opt.resume.endswith('safetensors'):
                ckpt = load_file(opt.resume, device='cpu')
            else:
                ckpt = torch.load(opt.resume, map_location='cpu')
            
            state_dict = model.state_dict()
            for k, v in ckpt.items():
                if k in state_dict: 
                    if state_dict[k].shape == v.shape:
                        state_dict[k].copy_(v)
            print("[RECON] Checkpoint loaded.")

        model.to(self.device)

        # 4. Dataset & Dataloader
        # Import inside function to avoid circular deps or heavy init at start if not needed
        if opt.data_mode == 's3':
            from core.provider_objaverse import MetaHumanDataset as Dataset
        else:
            raise NotImplementedError("Only 's3' data mode supported (MetaHumanDataset compatible structure).")

        train_dataset = Dataset(opt, training=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print("len(train_dataset) =", len(train_dataset))
        print("batch_size =", opt.batch_size)
        print("drop_last =", True)
        print("len(train_dataloader) =", len(train_dataloader))

        # 5. Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
        total_steps = opt.num_epochs * len(train_dataloader)
        pct_start = 300 / total_steps if total_steps > 300 else 0.1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(opt.mixed_precision != 'no'))

        # 6. Training Loop
        start_time = time.time()
        print(f"[RECON] Starting Training Loop ({opt.num_epochs} epochs)...")
        
        for epoch in range(opt.num_epochs):
            model.train()
            epoch_loss = 0
            epoch_psnr = 0
            
            for i, data in enumerate(train_dataloader):
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=(opt.mixed_precision != 'no')):
                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                    out = model(data, step_ratio)
                    loss = out['loss']
                    psnr = out['psnr']
                    loss = loss / opt.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % opt.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                current_loss = loss.detach() * opt.gradient_accumulation_steps
                epoch_loss += current_loss
                epoch_psnr += psnr.detach()

                # Logging (Simplified for merged pipeline)
                if i % 50 == 0:
                     print(f"[RECON] Epoch {epoch} | Step {i} | Loss: {current_loss.item():.6f}")

            # Optional: Save intermediate visualization logic here (omitted for brevity)
            
        total_seconds = time.time() - start_time
        print(f"[RECON] Finished. Time: {time.strftime('%H:%M:%S', time.gmtime(total_seconds))} results are saved in {opt.subject}.")

        # added: save time result to json
        timing_json_path = os.path.join(opt.subject, "timing.json")
        timing_info = {
            "recon_seconds": total_seconds,
            "recon_hms": time.strftime('%H:%M:%S', time.gmtime(total_seconds))
        }
        with open(timing_json_path, "w", encoding="utf-8") as f:
            json.dump(timing_info, f, indent=2, ensure_ascii=False)

        print(f"[RECON] Timing saved to {timing_json_path}")

        # Clean up to save VRAM for next iteration if multiple images
        del model, optimizer, scheduler, train_dataloader, train_dataset
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    
    if args.seed is not None:
        seed_everything(args.seed)
    else:
        seed_everything(random.randint(0, 65535))
    
    # Initialize Unified Pipeline
    pipeline = UnifiedPipeline(args)
    
    # Prepare Inputs
    if os.path.isdir(args.input_path):
        img_list = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_list = [args.input_path]

    # Main Loop
    for img_path in img_list:
        try:
            # Step 1: Generate Multi-view
            subject_dir = pipeline.run_mv_generation(img_path)
            
            # Step 2: Reconstruct 3D (LGM)
            pipeline.run_reconstruction(subject_dir)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            import traceback
            traceback.print_exc()