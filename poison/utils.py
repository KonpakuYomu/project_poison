import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

def img2tensor(image):
    w, h = image.size
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def tensor2img(tensor):
    to_pil = T.ToPILImage()
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    img = to_pil(tensor[0]).convert("RGB")
    return img
