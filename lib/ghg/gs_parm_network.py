import torch
from torch import nn
import torch.nn.functional as F
from core.extractor import UnetExtractor, ResidualBlock
import pdb
# from lib.ghg.network_module import Conv2dLayer

from lib.ghg.network_module import Conv2dLayer




class GSRegressor(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim


        depth_dim = 3*4
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + 3*5 + 3*4,
                                  self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4*5, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1*5, kernel_size=1),
            nn.Sigmoid()
        )
        self.feat_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )

    def forward(self, img, depth, img_feat):


        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)


        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)


        up3 = self.decoder3(feat3)
        up3 = self.up(up3)


        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)

        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
        up1 = self.up(up1)


        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)
        
        
        
        
        # print(f"SHARD BLOCK FEATURE SHAPE{out.shape}")
        
        #feat head 
        feat_out = self.feat_head(out)
        

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)
        
        # return rot_out, scale_out, opacity_out
        return rot_out, scale_out, opacity_out, 
    
    
class GSRegressorMC(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim


        depth_dim = 3*4
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + 3*5 + 3*4,
                                  self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4*5, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1*5, kernel_size=1),
            nn.Sigmoid()
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Tanh() #for -1,1 to match "img' 
        )

    def forward(self, img, depth, img_feat):

        img_feat1, img_feat2, img_feat3 = img_feat        
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)
        

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)


        up3 = self.decoder3(feat3)
        up3 = self.up(up3)


        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)

        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
        up1 = self.up(up1)


        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)
        
        # print(f"SHARD BLOCK FEATURE SHAPE{out.shape}")
        
        #color head 
        color_out = self.color_head(out)
        
        

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)
        
        # return rot_out, scale_out, opacity_out, color_out       
        return rot_out, scale_out, opacity_out, color_out
    
    
    
   
class GSRegressorMC_Depth(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim


        depth_dim = 3*4
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + 3*5 + 3*4,
                                  self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4*5, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1*5, kernel_size=1),
            nn.Sigmoid()
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Tanh() #for -1,1 to match "img' 
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            # nn.Tanh()
            
        )
        

    def forward(self, img, depth, img_feat):

        img_feat1, img_feat2, img_feat3 = img_feat        
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)
        

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)


        up3 = self.decoder3(feat3)
        up3 = self.up(up3)


        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)

        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
        up1 = self.up(up1)


        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)
        
        # print(f"SHARD BLOCK FEATURE SHAPE{out.shape}")
        
        #color head 
        color_out = self.color_head(out)
        
        

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)
        
        # depth head
        depth_out = self.depth_head(out)
        
        
        
        # return rot_out, scale_out, opacity_out, color_out       
        return rot_out, scale_out, opacity_out, color_out, depth_out


class GSRegressorDouble(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim


        depth_dim = 3*4
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + 3*5 + 3*4,
                                  self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4*5, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1*5, kernel_size=1),
            nn.Sigmoid()
        )
    
        # Define a wrapper function for the decoder
    def apply_decoder(self,decoder, x):
        for module in decoder.children():
            x = module(x)
        return x

    def forward(self, img, depth, img_feat):


        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)


        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)
        
        


        up3 = self.decoder3(feat3)
        up3 = self.up(up3)


        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)

        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
        up1 = self.up(up1)


        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)
        
        
        #INPUT IMAGE 
        input_up3 = self.apply_decoder(self.decoder3, feat3)
        input_up3 = self.up(input_up3)


        input_up2 = self.apply_decoder(self.decoder2, torch.cat([input_up3, feat2], dim=1))
        input_up2 = self.up(input_up2)

        input_up1 = self.apply_decoder(self.decoder1, torch.cat([input_up2, feat1], dim=1))
        input_up1 = self.up(input_up1)


        input_out = torch.cat([input_up1, img, depth], dim=1)
        input_out = self.out_conv(input_out)
        input_out = self.out_relu(input_out)
        
         # rot head
        input_rot_out = self.rot_head(input_out)
        input_rot_out = torch.nn.functional.normalize(input_rot_out, dim=1)

        # scale head
        input_scale_out = torch.clamp_max(self.scale_head(input_out), 0.01)

        # opacity head
        input_opacity_out = self.opacity_head(input_out)
        
        
        
        
        # print(f"SHARD BLOCK FEATURE SHAPE{out.shape}")
        
        
        
        

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)
        
        # return rot_out, scale_out, opacity_out
        return rot_out, scale_out, opacity_out, input_rot_out, input_scale_out, input_opacity_out


class GSRegressorSplit(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim


        depth_dim = 3*4
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        
        self.input_decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.input_decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.input_decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + 3*5 + 3*4,
                                  self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4*5, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1*5, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.input_rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4*5, kernel_size=1),
        )
        self.input_scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3*5, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.input_opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1*5, kernel_size=1),
            nn.Sigmoid()
        )
    
        # Define a wrapper function for the decoder
    def apply_decoder(self,decoder, x):
        for module in decoder.children():
            x = module(x)
        return x

    def forward(self, img, depth, img_feat):


        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)


        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)
        
        


        up3 = self.decoder3(feat3)
        up3 = self.up(up3)


        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)

        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
        up1 = self.up(up1)


        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)
        
        
        #INPUT IMAGE 
        input_up3 = self.input_decoder3(feat3)
        input_up3 = self.up(input_up3)


        input_up2 = self.input_decoder2(torch.cat([input_up3, feat2], dim=1))
        input_up2 = self.up(input_up2)

        input_up1 = self.input_decoder1(torch.cat([input_up2, feat1], dim=1))
        input_up1 = self.up(input_up1)


        input_out = torch.cat([input_up1, img, depth], dim=1)
        input_out = self.out_conv(input_out)
        input_out = self.out_relu(input_out)
        
         # rot head
        input_rot_out = self.input_rot_head(input_out)
        input_rot_out = torch.nn.functional.normalize(input_rot_out, dim=1)

        # scale head
        input_scale_out = torch.clamp_max(self.input_scale_head(input_out), 0.01)

        # opacity head
        input_opacity_out = self.input_opacity_head(input_out)
        
        
        
        
        # print(f"SHARD BLOCK FEATURE SHAPE{out.shape}")
        
        
        
        

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)
        
        # return rot_out, scale_out, opacity_out
        return rot_out, scale_out, opacity_out, input_rot_out, input_scale_out, input_opacity_out




class AuxRenderNet(nn.Module):
        """
        Auxiliary U-Net module that processes an input image of shape (B, 3, 1024, 1024)
        and produces an output of the same shape.
        
        The encoder uses the provided UnetExtractor with encoder dimensions [32, 48, 96].
        The decoder uses bilinear upsampling and skip connections.
        
        Assumption: The UnetExtractor is modified so that its initial convolution uses stride=1,
        yielding:
        - e1: (B, 32, 1024, 1024)
        - e2: (B, 48, 512, 512)
        - e3: (B, 96, 256, 256)
        """
        def __init__(self):
            super(AuxRenderNet, self).__init__()
            # Instantiate the encoder. We want our encoder dims to be [32, 48, 96].
            self.encoder = UnetExtractor(in_channel=3, encoder_dim=[32, 48, 96], norm_fn='group',initial_stride=1)
            
            # Build a decoder that upsamples back to 1024×1024.
            # Stage 1: Upsample e3 (256×256) to 512×512 and combine with e2.
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec1 = nn.Sequential(
                # Concatenated channels: 96 (upsampled e3) + 48 (e2) = 144.
                nn.Conv2d(144, 48, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            # Stage 2: Upsample result (512×512) to 1024×1024 and combine with e1.
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec2 = nn.Sequential(
                # Concatenated channels: 48 (upsampled from dec1) + 32 (e1) = 80.
                nn.Conv2d(80, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            # Final output layer: reduce from 32 channels to 3 (RGB).
            self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
        def forward(self, x):
            # x: (B, 3, 1024, 1024)
            # Encoder: get skip features
            e1, e2, e3 = self.encoder(x)  # e1: (B,32,1024,1024), e2: (B,48,512,512), e3: (B,96,256,256)
            
            # Decoder Stage 1: Upsample e3 to match e2's resolution.
            d1 = self.up1(e3)           # (B,96,512,512)
            # Concatenate with encoder output e2.
            d1_cat = torch.cat([d1, e2], dim=1)  # (B,96+48=144,512,512)
            d1_out = self.dec1(d1_cat)    # (B,48,512,512)
            
            # Decoder Stage 2: Upsample to match e1's resolution.
            d2 = self.up2(d1_out)         # (B,48,1024,1024)
            d2_cat = torch.cat([d2, e1], dim=1)  # (B,48+32=80,1024,1024)
            d2_out = self.dec2(d2_cat)    # (B,32,1024,1024)
            
            # Final output layer to get the desired 3-channel image.
            out = self.out_conv(d2_out)   # (B,3,1024,1024)
            out = torch.sigmoid(out)  
            return out



class AuxRender512(nn.Module):
    """
    A more memory-efficient U-Net for 1024×1024 image reconstruction:
      1) Immediately downsample the 1024×1024 input to 512×512.
      2) Continue downsampling to 256×256, 128×128 for the encoder.
      3) Upsample back to 1024×1024 in the decoder.
      4) Optionally concatenate the original full-res input at the last step
         to preserve high-frequency details.
    """
    def __init__(self):
        super().__init__()
        # --- ENCODER ---
        # Initial downsample: 3 -> 16 channels, stride=2 => from 1024×1024 to 512×512
        self.in_down = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(inplace=True)
        )
        
        # Encoder block 1 (512×512)
        self.enc1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(inplace=True),
        )
        
        # Downsample to 256×256
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(inplace=True),
        )
        
        # Encoder block 2 (256×256)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(inplace=True),
        )
        
        # Downsample to 128×128
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=128),
            nn.ReLU(inplace=True),
        )
        
        # Encoder block 3 (128×128)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
        )
        
        # --- DECODER ---
        # Up from 128×128 to 256×256
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(inplace=True),
        )
        
        # Up from 256×256 to 512×512
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(inplace=True),
        )
        
        # Up from 512×512 to 1024×1024
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Here we optionally skip‐connect the original input (3 channels).
        # So the final concat has 32 + 16 + 3 = 51 channels if we skip the in_down output,
        # or 32 + 3 = 35 channels if we skip the raw input x. 
        # We'll skip the raw input x to preserve the highest frequency details:
        self.dec0 = nn.Sequential(
            nn.Conv2d(32 + 3, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(inplace=True),
        )
        
        # Final output 16 -> 3 channels
        self.out_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (B, 3, 1024, 1024) input image
        Returns: (B, 3, 1024, 1024) reconstructed image
        """
        # --- ENCODER ---
        # 1) Downsample to 512×512
        x0 = self.in_down(x)       # (B, 16, 512, 512)
        e1 = self.enc1(x0)         # (B, 32, 512, 512)
        
        # 2) Downsample to 256×256
        d1 = self.down1(e1)        # (B, 64, 256, 256)
        e2 = self.enc2(d1)         # (B, 64, 256, 256)
        
        # 3) Downsample to 128×128
        d2 = self.down2(e2)        # (B, 128,128,128)
        e3 = self.enc3(d2)         # (B, 128,128,128)
        
        # --- DECODER ---
        # Up from 128×128 -> 256×256, skip with e2
        u2 = self.up2(e3)          # (B, 128,256,256)
        cat2 = torch.cat([u2, e2], dim=1)  # (B, 128+64=192, 256,256)
        d2_out = self.dec2(cat2)    # (B, 64, 256,256)
        
        # Up from 256×256 -> 512×512, skip with e1
        u1 = self.up1(d2_out)      # (B, 64, 512,512)
        cat1 = torch.cat([u1, e1], dim=1)  # (B, 64+32=96, 512,512)
        d1_out = self.dec1(cat1)    # (B, 32, 512,512)
        
        # Up from 512×512 -> 1024×1024
        u0 = self.up0(d1_out)      # (B, 32, 1024,1024)
        
        # Optionally skip with the *original* full-res input to preserve detail
        cat0 = torch.cat([u0, x], dim=1)  # (B, 32+3=35, 1024,1024)
        d0_out = self.dec0(cat0)         # (B, 16, 1024,1024)
        
        out = self.out_conv(d0_out)      # (B, 3, 1024,1024)
        return torch.tanh(out)
    


# Assuming ResidualBlock is defined elsewhere (as in your provided code).

class AuxSimple(nn.Module):
    """
    A lightweight reconstruction network with an extra skip connection.
    
    Accepts an input of shape [C, H, W] (e.g. [3, 1024, 1024]) and returns
    an output of the same shape. The model applies an initial convolution to lift
    the input to a small number of channels, then processes it with a few residual blocks,
    and finally adds a skip connection from the initial features before mapping back to 3 channels.
    
    Architecture:
      1. initial_conv: Lift 3 channels to base_channels.
      2. res_blocks: A series of residual blocks at full resolution.
      3. Global skip: Add the initial lifted features to the residual output.
      4. final_conv: Map features back to 3 channels, followed by tanh.
    """
    def __init__(self, num_residuals=3, base_channels=16):
        super().__init__()
        # Initial convolution: 3 -> base_channels.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels),
            nn.ReLU(inplace=True)
        )
        # Residual blocks operating at full resolution.
        res_blocks = [ResidualBlock(base_channels, base_channels, norm_fn='group')
                      for _ in range(num_residuals)]
        self.res_blocks = nn.Sequential(*res_blocks)
        # Final convolution: base_channels -> 3.
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)
        # Learnable scale for skip connection, initialized to 1.0
        self.skip_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # x is expected to have shape [C, H, W]. Add a dummy batch dimension.
        x = x.unsqueeze(0)  # [1, C, H, W]
        
        init_feat = self.initial_conv(x)         # [1, base_channels, H, W]
        res_out = self.res_blocks(init_feat)       # [1, base_channels, H, W]
        
        # Global skip: add the initial features to the residual output.
        out = init_feat + self.skip_scale * res_out
        out = self.final_conv(out)                # [1, 3, H, W]
        out = torch.tanh(out)                     # Constrain output to [-1, 1]
        
        out = out.squeeze(0)                      # Remove dummy batch dim -> [3, H, W]
        return out
    




# Define a wrapper function for the decoder
def apply_decoder(decoder, x):
    for module in decoder.children():
        x = module(x)
    return x



from torchvision.transforms.functional import crop
class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model_name="dinov2_vitl14_reg"):
        super().__init__()
        
        # Load the pretrained DinoV2 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        
       
        # Freeze most of the backbone but keep the last n blocks trainable
        # self._freeze_except_last_n_blocks(unfreeze_last_n_blocks)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Define the patch size and dimensions
        self.patch_size = 14
        self.embed_dim = 384  # ViT-L embedding dimension
        
        # Dimension reduction layer (trainable)
        # Define convolution layers for the three stages:
        # Stage 1: Reduce from 384 to 96 channels.
        self.stage1_conv = nn.Conv2d(self.embed_dim, 96, kernel_size=1)
        # Stage 2: Reduce from 96 to 48 channels.
        self.stage2_conv = nn.Conv2d(96, 48, kernel_size=1)
        # Stage 3: Reduce from 48 to 32 channels.
        self.stage3_conv = nn.Conv2d(48, 32, kernel_size=1)
        
        
    def _freeze_except_last_n_blocks(self, n):
        """Freeze all layers except the last n transformer blocks"""
        # Freeze patch embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze position embedding
        self.backbone.pos_embed.requires_grad = False
        
        # Freeze cls token
        if hasattr(self.backbone, 'cls_token'):
            self.backbone.cls_token.requires_grad = False
        
        # Freeze all blocks except the last n blocks
        total_blocks = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            if i >= total_blocks - n:
                # Keep these blocks trainable
                for param in block.parameters():
                    param.requires_grad = True
            else:
                # Freeze earlier blocks
                for param in block.parameters():
                    param.requires_grad = False
        
        # Keep norm layer trainable
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Input: x of shape [B, C, H, W]
        Output: features of shape [B, embed_dim//32, H//patch_size//2, W//patch_size//2]
        """
        x = crop(x, 0, 0, 1022, 1022)
        
        
        B, C, H, W = x.shape
        
        # Get patch tokens
        # Note: we're no longer using torch.no_grad() since we want gradients to flow
        # Forward pass through the backbone
        features = self.backbone.forward_features(x)
        
        # Get the patch tokens (excluding CLS token)
        patch_tokens = features["x_norm_patchtokens"]  # [B, num_patches, embed_dim]
        
        # Reshape patch tokens to spatial dimensions
        num_patches_side = int((H // self.patch_size))
        patch_tokens = patch_tokens.reshape(B, num_patches_side, num_patches_side, self.embed_dim)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # [B, embed_dim, H/patch_size, W/patch_size]
        
        merged_features = patch_tokens.sum(dim=0, keepdim=True) 
        
        # Stage 1: Reduce channels to 96 and upsample to 128×128.
        state1 = self.stage1_conv(merged_features)  # [1, 96, 73, 73]
        state1 = F.interpolate(state1, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Stage 2: Further reduce channels to 48 and upsample to 256×256.
        state2 = self.stage2_conv(state1)  # [1, 48, 128, 128]
        state2 = F.interpolate(state2, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Stage 3: Further reduce channels to 32 and upsample to 512×512.
        state3 = self.stage3_conv(state2)  # [1, 32, 256, 256]
        state3 = F.interpolate(state3, size=(512, 512), mode='bilinear', align_corners=False)
        
        return [state3, state2, state1]

        
        
        


# def extract_features(images_tensor, model_name="dinov2_vitl14_reg", unfreeze_last_n_blocks=1, training=True):
#     """
#     Extract features from a batch of images using DinoV2
    
#     Args:
#         images_tensor: Tensor of shape [B, C, H, W]
#         model_name: DinoV2 model variant
#         unfreeze_last_n_blocks: Number of transformer blocks to keep trainable
#         training: Whether to set the model in training mode
    
#     Returns:
#         features: Tensor of shape [B, 32, 512, 512]
#     """
#     device = images_tensor.device
    
#     # Initialize the feature extractor
#     feature_extractor = DinoV2FeatureExtractor(
#         model_name=model_name,
#         unfreeze_last_n_blocks=unfreeze_last_n_blocks
#     ).to(device)
    
#     # Set the model to training or evaluation mode
#     if training:
#         feature_extractor.train()
#     else:
#         feature_extractor.eval()
    
#     # Extract features
#     high, mid, low = feature_extractor(images_tensor)
    
#     return [low, mid, high]


# # Example usage
# if __name__ == "__main__":
#     # Create a dummy tensor of 3 images
#     batch_size = 3
#     images = torch.randn(batch_size, 3, 1024, 1024)
    
#     # Extract features in training mode
#     features, model = extract_features(images, unfreeze_last_n_blocks=2, training=True)
    
#     # Print the shape of the features
#     print(f"Input shape: {images.shape}")
#     print(f"Output features shape: {features.shape}")
    
#     # Print number of trainable parameters
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of trainable parameters: {trainable_params}")
    
#     # Example of using this in a training loop
#     optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    
#     # In your training loop
#     optimizer.zero_grad()
#     # Forward pass would already be done above
#     # loss = some_loss_function(features, targets)
#     # loss.backward()
#     # optimizer.step()