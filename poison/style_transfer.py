import os
import torch
import argparse

from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

def main(args):
    # load style transfer model
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_dir,
        revision="fp16",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    pipe_img2img.enable_xformers_memory_efficient_attention()

    # start style transfer
    clean_dir = f"{args.artist_dir}/train_clean"
    metadata_path = f"{clean_dir}/metadata.jsonl"
    if not os.path.exists(metadata_path):
        print(f"{clean_dir} is not a train set")
        return 
    trans_dir = f"{args.artist_dir}/trans_{args.model_type}"
    os.makedirs(trans_dir, exist_ok=True)
    
    for file_name in os.listdir(clean_dir):
        image_path = f"{clean_dir}/{file_name}"
        if not image_path.endswith(".jpg"):
            continue
        prompt = f"{args.aim_style} style"
        image = Image.open(f"{clean_dir}/{file_name}").convert("RGB")
        trans_image = pipe_img2img(prompt=prompt, image=image, strength=args.strength,
                                   guidance_scale=args.guidance, num_inference_steps=args.diff_steps).images[0]
        trans_image.save(f"{trans_dir}/{file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='style tranfer')
    parser.add_argument('--artist_dir', default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis', type=str)
    parser.add_argument('--model_dir', default='/root/autodl-tmp/model/stable-diffusion-2-1-base', type=str)
    parser.add_argument('--model_type',default="SD2.1", type=str)
    parser.add_argument('--aim_style', default='Cubism', type=str)
    parser.add_argument('--strength', default=0.4, type=float, help='learning rate.')
    parser.add_argument('--guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--diff_steps', default=50, type=int, help='learning rate.')
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
    