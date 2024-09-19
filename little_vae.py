import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.to_q = nn.Linear(in_channels,in_channels)
        self.to_k = nn.Linear(in_channels,in_channels)
        self.to_v = nn.Linear(in_channels,in_channels)
        self.groupnorm = nn.GroupNorm(32,in_channels, eps=1e-6, affine=True)
        self.scale = in_channels**-0.5
        self.linear_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        residual = x
        x = x.view(batch_size, channels, height*width).transpose(1,2)
        x = self.groupnorm(x.transpose(1, 2)).transpose(1, 2)

        # Query, Key, Value를 계산합니다.
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        # Attention Map을 계산합니다.
        energy = torch.bmm(Q, K.transpose(-1, -2)) * self.scale  # (batch_size, height*width, height*width)
        attention = energy.softmax(dim=-1)
        attention = torch.bmm(attention, V)

        # Linaer proj, residual
        out = self.linear_proj(attention).view(batch_size, channels, height, width)
        out = out + residual
        return out
    
    def load_vae_weight(self, vae_attention): # vae_attention = vae.decoder.mid_block.attentions[0].
        self.to_q.weight = vae_attention.to_q.weight
        self.to_k.weight = vae_attention.to_k.weight
        self.to_v.weight = vae_attention.to_v.weight
        self.groupnorm.weight = vae_attention.group_norm.weight
        self.linear_proj.weight = vae_attention.to_out[0].weight


class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32,in_channels, eps=1e-6, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.norm2 = nn.GroupNorm(32,in_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = x + residual

        return x

    def load_vae_weight(self, vae_resnet): #vae_resnet = vae.decoder.mid_block.resnets[0], [1]
        self.norm1.weight = vae_resnet.norm1.weight
        self.norm2.weight = vae_resnet.norm2.weight
        self.conv1.weight = vae_resnet.conv1.weight
        self.conv2.weight = vae_resnet.conv2.weight
        

# little vae structure definition
class little_vae(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(little_vae, self).__init__()  

        self.post_quant_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1), stride=(1,1))

        # From <AutoEncoderKL> decoder
        self.vae_magic = 0.18215
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.Attnblock_decoder = SelfAttention(out_channels)
        self.Resblock0_decoder = ResNetBlock(out_channels)
        self.Resblock1_decoder = ResNetBlock(out_channels)


        # From <AutoencoderKl> encoder
        self.Attnblock_encoder = SelfAttention(out_channels)
        self.Resblock0_encoder = ResNetBlock(out_channels)
        self.Resblock1_encoder = ResNetBlock(out_channels)
        self.conv_norm_out = nn.GroupNorm(32,out_channels, eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 내가 바꿈
        # self.quant_conv = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3,3), stride=(1,1)) 2개중에 하나의 방법을 사용하면 될 듯?

        self.quant_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1,1), stride=(1,1))
    # self.quant_conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1,1), stride=(1,1)) # 내가 바꿈

    def load_vae_weight(self, vae):
        #load vae part
        self.post_quant_conv.weight = vae.post_quant_conv.weight

        #load decoder parts
        self.conv_in.weight = vae.decoder.conv_in.weight
        self.Attnblock_decoder.load_vae_weight(vae.decoder.mid_block.attentions[0])
        self.Resblock0_decoder.load_vae_weight(vae.decoder.mid_block.resnets[0])
        self.Resblock1_decoder.load_vae_weight(vae.decoder.mid_block.resnets[1])

        self.Attnblock_encoder.load_vae_weight(vae.encoder.mid_block.attentions[0])
        self.Resblock0_encoder.load_vae_weight(vae.encoder.mid_block.resnets[0])
        self.Resblock1_encoder.load_vae_weight(vae.encoder.mid_block.resnets[1])

        #load encoder part
        self.conv_norm_out.weight = vae.encoder.conv_norm_out.weight
        self.conv_out.weight = vae.encoder.conv_out.weight
        self.quant_conv.weight = vae.quant_conv.weight
        
    def forward(self, x):
        z = x / self.vae_magic
        z = self.post_quant_conv(z)     # [1,4,64,64] -> [1,4,64,64]
        z = self.conv_in(z)             # [1,4,64,64] -> [1,512,64,64]

        # Decoder
        # 1. First Conv block
        z = self.Resblock0_decoder(z)   # [1,512,64,64] -> [1,512,64,64]
        
        # 2. Attention block
        z = self.Attnblock_decoder(z)   # [1,512,64,64] -> [1,512,64,64]

        # 3. Second Conv block
        z = self.Resblock1_decoder(z)   # [1,512,64,64] -> [1,512,64,64]
        # Encoder
        # 1. First Conv block
        z = self.Resblock0_encoder(z)   # [1,512,64,64] -> [1,512,64,64]
        
        # 2. Attention block
        z = self.Attnblock_encoder(z)   # [1,512,64,64] -> [1,512,64,64]

        # 3. Second Conv block
        z = self.Resblock1_encoder(z)   # [1,512,64,64] -> [1,512,64,64]

        z = self.conv_norm_out(z)       # [1,512,64,64] -> [1,512,64,64]
        z = self.conv_act(z)
        z = self.conv_out(z)            # [1,512,64,64] -> [1,8,64,64]
        parameters = self.quant_conv(z) # shape: [1,8,64,64] / [1,4,64,64]
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        eps = torch.randn(self.mean.shape).to(device=self.mean.device)
        x = self.mean + self.std * eps

        output = x * self.vae_magic
        # output = parameters   # 내가 바꿈

        return output

    def save(self, path):
        ckpt = {'state_dict': self.state_dict()}
        torch.save(ckpt, path)

    def load(self, ckpt):
        self.load_state_dict(ckpt['state_dict'])

if __name__ == '__main__':
    # Model instantiation
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    model = little_vae(in_channels=4, out_channels=512)
    print(model)

    # Test input data
    input_data = torch.randn(1, 4, 64, 64)
    model.load_vae_weight(vae)
    # Model test
    output = model(input_data)
    print("Output Size:", output.size())  # [1, 4, 64, 64]
