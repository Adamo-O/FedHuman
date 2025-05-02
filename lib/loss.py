
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    valid = (valid >= 0.5)
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train_epe': epe.mean().item(),
        'train_1px': (epe < 1).float().mean().item(),
        'train_3px': (epe < 3).float().mean().item()
    }

    return flow_loss, metrics


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=['relu2_2'], weight=0.1):
        """
        layers: list of layer names at which to compute the loss.
        weight: scaling factor for the perceptual loss.
        """
        super(VGGPerceptualLoss, self).__init__()
        self.weight = weight

        # Load a pretrained VGG19 network
        vgg_full = models.vgg19(weights=None)
        vgg_full.load_state_dict(torch.load("weights/vgg19-dcbb9e9d.pth", map_location="cpu"))
        vgg = vgg_full.features
        
        # Dictionary to map layer indices to names (this example uses a common mapping)
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '17': "relu3_4",
            '26': "relu4_4"
        }
        self.layers = layers

        # Extract only the layers we need
        self.vgg_layers = vgg.eval()  # set to evaluation mode
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # Define the normalization transform (VGG expects images normalized to ImageNet stats)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        """
        x: reconstructed image tensor of shape (B, 3, H, W)
        y: ground truth image tensor of shape (B, 3, H, W)
        """
        # Ensure that input images are in [0,1] and then normalize
        # If your network already outputs in [0,1], apply normalization channel-wise.
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0.0
        # Run the images through the VGG network and accumulate losses
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            y = module(y)
            layer_name = self.layer_name_mapping.get(name, None)
            if layer_name in self.layers:
                loss += F.l1_loss(x, y)
        return self.weight * loss
    
    

def laplacian_kernel(channels, device):
    # Define a simple 3x3 Laplacian kernel.
    kernel = torch.tensor([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=torch.float32, device=device)
    # Expand kernel dimensions for depthwise convolution:
    # Shape: [channels, 1, 3, 3]
    kernel = kernel.expand(channels, 1, 3, 3)
    return kernel

def laplacian_loss(pred, gt):
    """
    Computes the Laplacian loss between pred and gt.
    Args:
        pred (Tensor): Predicted images, shape [B, C, H, W]
        gt (Tensor): Ground truth images, shape [B, C, H, W]
    Returns:
        loss (Tensor): A scalar loss value.
    """
    pred =  pred.unsqueeze(0)
    gt  =    gt.unsqueeze(0)
    channels = pred.shape[1]
    kernel = laplacian_kernel(channels, pred.device)
    
    # Compute Laplacian responses via convolution.
    lap_pred = F.conv2d(pred, kernel, padding=1, groups=channels)
    lap_gt   = F.conv2d(gt, kernel, padding=1, groups=channels)
    
    # Compute L1 loss between the Laplacian responses.
    loss = F.l1_loss(lap_pred, lap_gt)
    return loss

def total_variation_loss(img):
    """
    img: (B, C, H, W) tensor
    weight: scaling factor for the TV term
    returns a scalar tensor for the TV loss
    """
    # Shift left/right and top/bottom by 1 pixel and compute difference
    diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    tv = (diff_h + diff_w)
    return tv


from lib.lpips import LPIPS  # make sure you have this module

class LPIPSLoss(nn.Module):
    def __init__(self, net='vgg',version='0.1'):
        super(LPIPSLoss, self).__init__()
        self.lpips = LPIPS(net=net,version=version)
        # Freeze the LPIPS network parameters
        for param in self.lpips.parameters():
            param.requires_grad = False
        self.lpips.cuda()

    def forward(self, x, y,scale):
        # Scale inputs from [0,1] to [-1,1]
        if scale:
            scale_for_lpips = lambda x: 2 * x - 1
            x = scale_for_lpips(x)
            y = scale_for_lpips(y)
        return self.lpips(x, y).mean()