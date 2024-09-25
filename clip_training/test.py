import os
import csv
import json
import torch
import argparse
import open_clip
from PIL import Image

def main(args):
    # loading finetuned models
    checkpoint = f"{args.log_dir}/checkpoints/epoch_{args.epoch}.pt"
    tokenizer = open_clip.get_tokenizer(args.model_type)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_type, pretrained=checkpoint)
    model.to("cuda")
    model.eval()

    # get all output types (all styles)
    with open(args.image_info_path, 'r') as file:
        data = json.load(file)
        all_style = list(data)

    # predicting the style of test set images, save test result into info
    text = tokenizer(all_style).to("cuda")
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    with open(args.test_csv_path, 'r', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        test_num = len(reader)
        info = {}
        for style in all_style:
            info[style] = {"total_num": 0, "correct_prediction": 0}
        info["all_style"] = {"total_num": test_num, "correct_prediction": 0}
        
        counter = 0
        for row in reader:
            counter += 1
            if counter % 500 == 0:
                print(f"predicting...[{counter}/{test_num}]")
            image_path, style = row["image_path"], row["style"]
            image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda").half()              
    
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                _, top3_class_index = torch.sort(probs, descending=True)
                top3_class_index = top3_class_index[0][:3]
                top3_class_name = [all_style[k] for k in top3_class_index]
                if style in top3_class_name:
                    info[style]["correct_prediction"] += 1
                    info["all_style"]["correct_prediction"] += 1
                info[style]["total_num"] += 1

    for style in info:
        info[style]["accuracy"] = info[style]["correct_prediction"] / info[style]["total_num"]
    print(info)
    with open(f"{args.log_dir}/epoch{args.epoch}_test.json", 'w') as file:
        json.dump(info, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parquet extracting')
    parser.add_argument('--model_type', default='convnext_base_w', type=str)
    parser.add_argument('--log_dir', default='./open_clip/src/logs/convnext_base_w-lr_1e-05-b_96-j_8-p_amp', type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--image_info_path', default='/root/autodl-tmp/dataset/origin_images/info.json', type=str)
    parser.add_argument('--test_csv_path', default='/root/autodl-tmp/dataset/clip_training/test.csv', type=str)
    args = parser.parse_args()
    main(args)

