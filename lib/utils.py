
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import math
from PIL import Image
import os
import cv2

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def save_image(tensor, name, path="/home/s_pinon/p/ghg/debug/"):
    """
    Saves a torch tensor as an image file.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (1, 3, H, W).
        name (str): The name of the image file (without extension).
        path (str): The directory where the image will be saved.
    """
    # Remove the batch dimension (resulting shape: [3, H, W])
    tensor = tensor.squeeze(0)
    
    # Permute dimensions to get shape [H, W, 3]
    tensor = tensor.permute(1, 2, 0)
    
    # Convert tensor to numpy array
    np_img = tensor.cpu().numpy()
    
    # If pixel values are in [0,1], scale them to [0,255]
    if np_img.max() <= 1.0:
        np_img = (np_img * 255).astype(np.uint8)
    else:
        np_img = np_img.astype(np.uint8)
    
    # Create output directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Construct the full file path (ensuring .png extension)
    full_path = os.path.join(path, f"{name}.png")
    
    # Create and save the image
    image = Image.fromarray(np_img)
    image.save(full_path)

def bytes_to_human(nbytes):
    # Convert bytes into a human readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"

def print_memory_usage(label="Memory"):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{label} allocated: {bytes_to_human(allocated)}")
    print(f"{label} reserved: {bytes_to_human(reserved)}")
    
    
def annotate_image_with_psnr(combined_image, tmp_psnr,tmp_lpips=None, position=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2, auxilary=False):
    """
    Annotates the combined image with the PSNR value.

    Args:
        combined_image (numpy.ndarray): Input image as a numpy array. It can be in float (range 0-1 or -1 to 1 based on auxilary flag)
                                         or uint8 format.
        tmp_psnr (float): PSNR value to annotate onto the image.
        position (tuple): (x, y) position for the text. Default is (10, 30).
        font_scale (int, optional): Font scale for the text. Default is 1.
        color (tuple, optional): Text color in BGR format. Default is white (255, 255, 255).
        thickness (int, optional): Thickness of the text. Default is 2.
        auxilary (bool, optional): If True, assumes input is in range [-1, 1] (default is False, assuming [0, 1]).
    
    Returns:
        numpy.ndarray: The annotated image.
    """
    # Convert the image to uint8 if it's in float format.
    if combined_image.dtype != np.uint8:
        if auxilary:
            # Map from [-1, 1] to [0, 255]
            image_to_annotate = (((combined_image + 1) / 2) * 255).astype(np.uint8)
        else:
            # Map from [0, 1] to [0, 255]
            image_to_annotate = (combined_image * 255).astype(np.uint8)
    else:
        image_to_annotate = combined_image.copy()

    # Ensure the image is stored in contiguous memory.
    image_to_annotate = np.ascontiguousarray(image_to_annotate)

    # Prepare the text string.
    if tmp_lpips is not None:
        text = f"PSNR: {tmp_psnr:.2f} LPIPS: {tmp_lpips:.2f}"
    else:
        text = f"PSNR: {tmp_psnr:.2f}"
    
    # Overlay the text on the image.
    cv2.putText(image_to_annotate, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    
    return image_to_annotate