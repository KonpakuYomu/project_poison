import os
import torch
import argparse
import jsonlines

from PIL import Image
from diffusers import AutoencoderKL
from utils import img2tensor, tensor2img

def main(args):
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device)
    for name, param in vae.encoder.named_parameters():
        param.requires_grad = False
    for name, param in vae.decoder.named_parameters():
        param.requires_grad = False

    os.makedirs(args.save_dir, exist_ok=True)
    for file_name in os.listdir(args.image_dir):
        if not file_name.endswith(".png"):
            continue
        init_image = Image.open(f"{args.image_dir}/{file_name}").convert("RGB")
        x = img2tensor(init_image).to(device)
        x_emb = vae.encode(x).latent_dist.sample()
        x_dec = vae.decode(x_emb).sample
        decode_image = tensor2img(x_dec)
        decode_image.save(f"{args.save_dir}/{file_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='using VAE to encode and decode images')
    parser.add_argument('--model', default='/root/autodl-tmp/model/stable-diffusion-2-1-base', type=str)
    parser.add_argument('--image_dir', type=str, default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis/trans_SD2.1')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis/decode/trans_SD2.1')
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')
    parser.add_argument('--device', default='cuda:1', type=str, help='device used for training')
    
    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


