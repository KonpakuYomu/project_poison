import os
import csv
import json

import argparse
import torch
from PIL import Image
import open_clip

def test(args):
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_type, pretrained=args.model_path).to("cuda")
    tokenizer = open_clip.get_tokenizer(args.model_type)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

    text = tokenizer(["a diagram", "a dog", "a cat"]).to("cuda")
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    with open(args.test_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path, style = row["image_path"], row["style"]
            image = preprocess(Image.open(image_path)).unsqueeze(0)
    
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            print("Label probs:", text_probs)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parquet extracting')
    parser.add_argument('--model_type', default='convnext_base_w', type=str)
    parser.add_argument('--model_path', default='/root/autodl-tmp/dataset/wikiart', type=str)
    parser.add_argument('--test_csv_path', default='/root/autodl-tmp/dataset/clip_training/test.csv', type=str)
    args = parser.parse_args()
    main(args)

