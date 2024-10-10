import os
import csv
import json
import torch
import argparse
import open_clip
from PIL import Image

def main(args):
    # loading finetuned models
    tokenizer = open_clip.get_tokenizer(args.model_type)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_type, pretrained=args.clip_path)
    model.to("cuda")
    model.eval()

    # get all possibles types (all styles)
    with open(args.all_style_info, 'r') as file:
        data = json.load(file)
        all_style = list(data)

    text = tokenizer(all_style).to("cuda")
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    counter, shift_num, target_num = 0, 0, 0
    for file in os.listdir(args.test_dir):
        if not file.endswith(".jpg"):
            continue
        counter += 1
        image_path = f"{args.test_dir}/{file}"
        image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda").half()              
    
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, top3_class_index = torch.sort(probs, descending=True)
            top3_class_index = top3_class_index[0][:3]
            top3_class_name = [all_style[k] for k in top3_class_index]
            if not args.origin_style in top3_class_name:
                shift_num += 1
                if args.target_style in top3_class_name:
                    target_num += 1

    style_shift_rate = shift_num / counter
    target_rate = target_num / counter
    print(f"test data num:{counter}, style shift rate:{style_shift_rate*100}%, target rate:{target_rate*100}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parquet extracting')
    parser.add_argument('--model_type', default='convnext_base_w', type=str)
    parser.add_argument('--clip_path', default='/root/autodl-tmp/model/CLIP_style-convnext_base_w-wikiart-ep10-b96/epoch_10.pt', type=str)
    parser.add_argument('--test_dir', default='/root/autodl-tmp/dataset/sample_resolution_512/Abstract_Expressionism/sam-francis/generated/train_clean', type=str)
    parser.add_argument('--all_style_info', default='/root/autodl-tmp/dataset/origin_images/info.json')
    parser.add_argument('--origin_style', default='Abstract_Expressionism')
    parser.add_argument('--target_style', default='Cubism')
    args = parser.parse_args()
    main(args)

