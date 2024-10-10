import os
import torch
import shutil
import argparse
import jsonlines

from PIL import Image
from utils import img2tensor, tensor2img
from poison_core import glaze, poi
from diffusers import AutoencoderKL
    
def main(args):
    # load vae model
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device)
    for name, param in vae.encoder.named_parameters():
        param.requires_grad = False
    for name, param in vae.decoder.named_parameters():
        param.requires_grad = False

    clean_dir = f"{args.artist_dir}/train_clean"
    if not os.path.exists(f"{clean_dir}/metadata.jsonl"):
        print(f"{args.artist_dir} is not an artist directory")
        return
    trans_dir = f"{args.artist_dir}/trans_SD2.1"
    if args.poison_method == "poi":
        save_dir = f"{args.artist_dir}/poi/p1_{args.p1}_p2_{args.p2}_alpha{args.alpha}_lr{args.lr}_iter{args.iters}"
    else:
        save_dir = f"{args.artist_dir}/glaze/p{args.p1}_alpha{args.alpha}_lr{args.lr}_iter{args.iters}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(f"{clean_dir}/metadata.jsonl", f"{save_dir}/metadata.jsonl")
        
    # start poisoning
    with open(f"{clean_dir}/metadata.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            file_name = item['file_name']
            init_image = Image.open(f"{clean_dir}/{file_name}").convert("RGB")
            trans_image = Image.open(f"{trans_dir}/{file_name}").convert("RGB")
            x = img2tensor(init_image).to(device)
            x_t = img2tensor(trans_image).to(device)
            
            if args.poison_method == "poi":
                #x_adv, target = poi(x, x_t, model=vae, p1=args.p1, p2=args.p2, iters=args.iters)
                x_adv, target = poi(x, x_t, model=vae, p1=args.p1, p2=args.p2, alpha=args.alpha, iters=args.iters, lr=args.lr)
                target_dir = f"{save_dir}/target"
                os.makedirs(target_dir, exist_ok=True)
                target_image = tensor2img(target)
                target_image.save(f"{target_dir}/{file_name}")
            else:
                x_adv = glaze(x, x_t, model=vae.encode, p=args.p1, alpha=args.alpha, iters=args.iters, lr=args.lr)
            adv_image = tensor2img(x_adv)
            adv_image.save(f"{save_dir}/{file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='poisoning')
    parser.add_argument('--model', default='/root/autodl-tmp/model/stable-diffusion-2-1-base', type=str)
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--artist_dir', default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis', type=str)

    parser.add_argument('--poison_method', type=str, choices=['glaze', 'poi'], default='glaze')
    # mutual hyperparameters of poi and glaze
    parser.add_argument('--iters', default=200, type=int)
    parser.add_argument('--p1', default=0.05, type=float)
    # special hyperparameters of poi
    parser.add_argument('--p2', default=0.1, type=float)
    # special hyperparameters of glaze
    parser.add_argument('--alpha', default=30, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
 
    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


