import torch
import torch as th
import torch.nn as nn
import os
from typing import Any, Dict, List, Tuple, Union

from einops import rearrange
from sgm.models.diffusion import DiffusionEngine
from sgm.util import log_txt_as_img, exists, instantiate_from_config
from safetensors.torch import load_file as load_safetensors
from sgm.modules.autoencoding.lpips.loss.lpips import LPIPS
from .id_model import Backbone

class VideoLDM(DiffusionEngine):
    def __init__(self, num_samples, trained_param_keys=[''], *args, **kwargs):
        self.trained_param_keys = trained_param_keys
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.lpips = LPIPS().eval()
        self.id_model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.id_model.load_state_dict(torch.load("ckpts/trained_arcface.pth"))
        self.id_model.eval()

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
        elif path.endswith("pt"):
            sd_raw = torch.load(path, map_location="cpu")
            sd = {}
            for k in sd_raw['module']:
                sd[k[len('module.'):]] = sd_raw['module'][k]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        self.load_state_dict(sd, strict=False)
    
    @torch.no_grad()
    def add_custom_cond(self, batch, infer=False):
        batch['num_video_frames'] = self.num_samples
        
        image = batch['video'][:, :, 0]
        batch['cond_frames_without_noise'] = image.half()
        
        N = batch['video'].shape[0]
        if not infer:
            cond_aug = ((-3.0) + (0.5) * torch.randn((N,))).exp().cuda().half()
        else:
            cond_aug = torch.full((N, ), 0.02).cuda().half()
        batch['cond_aug'] = cond_aug
        batch['cond_frames'] = (image + rearrange(cond_aug, 'b -> b 1 1 1') * torch.randn_like(image)).half()
            
        # for dataset without indicator
        if not 'image_only_indicator' in batch:
            batch['image_only_indicator'] = torch.zeros((N, self.num_samples)).cuda().half()
        return batch

    def shared_step(self, batch: Dict) -> Any:
        frames = self.get_input(batch) # b c t h w
        batch = self.add_custom_cond(batch)
        
        frames_reshape = rearrange(frames, 'b c t h w -> (b t) c h w')
        x = self.encode_first_stage(frames_reshape)
        
        batch["global_step"] = self.global_step
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def forward(self, x, batch):
        loss1, model_output = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        output = self.decode_first_stage(model_output.half())
        target = self.decode_first_stage(x.half())
        with torch.no_grad():
            loss2 = self.lpips(output, target).reshape(-1)
            loss3 = self.id_loss(output, target)
        loss_mean = (loss1+loss2 + 1 - loss3).mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict
    
    def id_loss(self, output, target):
        output = (output + 1.) / 2.
        target = (target + 1.) / 2.
        output = output.clamp(0, 1)
        target = target.clamp(0, 1)
        output = torch.nn.functional.interpolate(output, size=(112, 112), mode='bilinear', align_corners=False)
        target = torch.nn.functional.interpolate(target, size=(112, 112), mode='bilinear', align_corners=False)
        emb_output = self.id_model(output)
        emb_target = self.id_model(target)
        loss = torch.einsum('ij,ij->i', emb_output, emb_target)
        return loss 

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        frames = self.get_input(batch)
        batch = self.add_custom_cond(batch, infer=True)
        N = min(frames.shape[0], N)
        frames = frames[:N]
        x = rearrange(frames, 'b c t h w -> (b t) c h w')
        
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )
        
        sampling_kwargs = {}

        aes = c['vector'][:, -256-256-256]
        cm1 = c['vector'][:, -256-256]
        cm2 = c['vector'][:, -256-192]
        cm3 = c['vector'][:, -256-128]
        cm4 = c['vector'][:, -256-64]
        caption =  batch['caption'][:N]
        for idx in range(N):
            sub_str = str(aes[idx].item()) + '\n' + str(cm1[idx].item()) + '\n' + str(cm2[idx].item()) + '\n' + str(cm3[idx].item()) + '\n' + str(cm4[idx].item())
            caption[idx] = sub_str + '\n' + caption[idx]
        
        x = x.to(self.device)
        
        z = self.encode_first_stage(x.half())
        x_rec = self.decode_first_stage(z.half())
        log["reconstructions-video"] = rearrange(x_rec, '(b t) c h w -> b c t h w', t=self.num_samples)
        log["conditioning"] = log_txt_as_img((512, 512), caption, size=16)
        
        for k in c:
            if isinstance(c[k], torch.Tensor):
                if k == 'concat':
                    c[k], uc[k] = map(lambda y: y[k][:N * self.num_samples].to(self.device), (c, uc))
                else:
                    c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            N * 2, self.num_samples
        ).to(self.device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
        def denoiser(input, sigma, c):
            return self.denoiser(
                self.model, input, sigma, c, **additional_model_inputs
            )
            
        if sample:
            with self.ema_scope("Plotting"):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    randn = torch.randn(z.shape, device=self.device)
                    samples = self.sampler(denoiser, randn, cond=c, uc=uc)
            samples = self.decode_first_stage(samples.half())
            log["samples-video"] = rearrange(samples, '(b t) c h w -> b c t h w', t=self.num_samples)
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        if 'all' in self.trained_param_keys:
            params = list(self.model.parameters())
        else:
            names = []
            params = []
            for name, param in self.model.named_parameters():
                flag = False
                for k in self.trained_param_keys:
                    if k in name:
                        names += [name]
                        params += [param]
                        flag = True
                    if flag:
                        break
            print(names)
             
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
                
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt
    

from sgm.modules.encoders.modules import AbstractEmbModel
from .id_model import Backbone
class ArcFaceEmbedder(AbstractEmbModel):
    def __init__(self,n_cond_frames=1):
        super().__init__()
        self.id_model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.id_model.load_state_dict(torch.load("ckpts/trained_arcface.pth"))
        self.id_model.eval()
        # 构建一个512到1024的mlp层
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.n_cond_frames = n_cond_frames

    def preprocess(self, frame):
        frame = (frame + 1.) / 2.
        frame = frame.clamp(0, 1)
        target = torch.nn.functional.interpolate(frame, size=(112, 112), mode='bilinear', align_corners=False)
        return target

    def forward(self, vid):
        with torch.no_grad():
            vid = self.preprocess(vid)
            id_embedding = self.id_model(vid)
        id_embedding = self.mlp(id_embedding)
        id_embedding = rearrange(id_embedding, "(b t) d -> b t d", t=self.n_cond_frames)
        return id_embedding
