import time
import os
import yaml
import torch
from core.options import Options
from core.models import LGM
from safetensors.torch import load_file
import kiui
import argparse
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image

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

def main(opt):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")


    model = LGM(opt)
    
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
                else:
                    print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                print(f'[WARN] unexpected param {k}: {v.shape}')
    
    model.to(device)

    if opt.data_mode == 's3':
        from core.provider_objaverse import MetaHumanDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 300 / total_steps if total_steps > 300 else 0.1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(opt.mixed_precision != 'no'))

    start_time = time.time()
    
    for epoch in range(opt.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        
        for i, data in enumerate(train_dataloader):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

            with torch.cuda.amp.autocast(enabled=(opt.mixed_precision != 'no')):
                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                # 考虑梯度累积
                loss = loss / opt.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % opt.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            current_loss = loss.detach() * opt.gradient_accumulation_steps
            epoch_loss += current_loss
            epoch_psnr += psnr.detach()

            if i % 50 == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()    
                print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {current_loss.item():.6f}")
                
                gt_images = data['images_output'].detach().cpu().numpy()
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                kiui.write_image(os.path.join(opt.workspace, f'train_gt_images_{epoch}_{i}.jpg'), gt_images)

                pred_images = out['images_pred'].detach().cpu().numpy()
                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                kiui.write_image(os.path.join(opt.workspace, f'train_pred_images_{epoch}_{i}.jpg'), pred_images)

        print(f"[train] epoch: {epoch} loss: {epoch_loss/len(train_dataloader):.6f} psnr: {epoch_psnr/len(train_dataloader):.4f}")

    # 6. 保存模型
    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"[train] Finished. Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_seconds))}")
    
    # 保存最终权重
    os.makedirs(opt.workspace, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(opt.workspace, 'model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_config', type=str, default="configs/recon/inference.yaml")
    parser.add_argument('--resume', type=str, default="ckpts/recon.safetensors")
    parser.add_argument('--image_path', type=str, default="results/1")
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=100)
    params = parser.parse_args()
    with open(params.recon_config, 'r') as f:
        recon_config = yaml.safe_load(f)
    params_recon = Options(**recon_config)
    params_recon.resume = params.resume
    params_recon.lr = params.lr
    params_recon.num_epochs = params.num_epochs

    high_stage_path = os.path.join(params.image_path, "high_stage")
    subject_list = []
    if os.path.isdir(high_stage_path):
        subject_list.append(params.image_path)
    else:
        for subject in os.listdir(params.image_path):
            subject_list.append(os.path.join(params.image_path, subject))

    for subject in tqdm(subject_list):
        params_recon.subject = subject
        print(f"dealing with subject {subject}...")
        image_path = os.path.join(subject, "high_stage")
        for image in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, image))
            if img.mode != "RGBA":
                img = preprocess(img)
                img.save(os.path.join(image_path, image))
        main(params_recon)