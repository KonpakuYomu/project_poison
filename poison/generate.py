import os
import json
import torch
import argparse

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel

def main(args):
    with open(args.test_prompt_path, 'r') as file:
        text_prompts = json.load(file)

    pipe_text2img = StableDiffusionPipeline.from_pretrained(args.model_dir, torch_dtype=torch.float16)
    pipe_text2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_text2img.scheduler.config)
    if args.checkpoint is not None:
        pipe_text2img.unet = UNet2DConditionModel.from_pretrained(
            f"{args.model_dir}/checkpoint-{args.checkpoint}", 
            subfolder="unet", torch_dtype=torch.float16
        )
    pipe_text2img = pipe_text2img.to(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for text in text_prompts:
        generated_image = pipe_text2img(prompt=text, num_inference_steps=args.diff_steps).images[0]
        generated_image.save(f"{args.output_dir}/{text}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image generation')
    parser.add_argument('--model_dir', type=str, 
                        default='/root/autodl-tmp/model/finetuned_SD2.1/Abstract_Expressionism/sam-francis/glaze/p0.05_alpha30_lr0.01_iter500')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--test_prompt_path', type=str, 
                        default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis/test/text.json')
    parser.add_argument('--output_dir', type=str, 
                        default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis/generated/glaze/p0.05_alpha30_lr0.01_iter500')
    parser.add_argument('--diff_steps', default=100, type=int, help='learning rate.')
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used for training')

    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


