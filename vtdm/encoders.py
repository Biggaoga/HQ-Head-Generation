import torch
from sgm.modules.encoders.modules import AbstractEmbModel
from tools.aes_score import MLP, normalized
import clip
from sgm.modules.diffusionmodules.util import timestep_embedding
from sgm.util import autocast
        

class AesEmbedder(AbstractEmbModel):
    def __init__(self, freeze=True):
        super().__init__()
        aesthetic_model, _ = clip.load("ckpts/ViT-L-14.pt")
        del aesthetic_model.transformer
        self.aesthetic_model = aesthetic_model
        self.aesthetic_mlp = MLP(768)
        self.aesthetic_mlp.load_state_dict(torch.load("ckpts/sac+logos+ava1-l14-linearMSE.pth"))

        if freeze:
            self.freeze()
        
    def freeze(self):
        self.aesthetic_model = self.aesthetic_model.eval()
        self.aesthetic_mlp = self.aesthetic_mlp.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @autocast
    @torch.no_grad()
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        y = x[:, :, T//2]
        y = torch.nn.functional.interpolate(y, [224, 384], mode='bilinear')
        y = y[:, :, :, 80:304]
        y = (y + 1) * 0.5
        y[:, 0] = (y[:, 0] - 0.48145466) / 0.26862954
        y[:, 1] = (y[:, 1] - 0.4578275) / 0.26130258
        y[:, 2] = (y[:, 2] - 0.40821073) / 0.27577711
        
        image_features = self.aesthetic_model.encode_image(y)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        aesthetic = self.aesthetic_mlp(torch.from_numpy(im_emb_arr).to('cuda').type(torch.cuda.FloatTensor))
        
        return torch.cat([aesthetic, timestep_embedding(aesthetic[:, 0] * 100, 255)], 1)