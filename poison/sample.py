import os
import json
import random
import argparse
import jsonlines

from PIL import Image
from random import sample
from transformers import BlipProcessor, BlipForConditionalGeneration

import torchvision.transforms as T


def main(args):
    processor = BlipProcessor.from_pretrained(args.blip_model)
    model = BlipForConditionalGeneration.from_pretrained(args.blip_model).to("cuda")    
    train_num, test_num = 128, 100
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    with open(f"{args.origin_images_dir}/info.json", 'r') as file:
        image_info = json.load(file)

    for style in image_info:
        for artist in image_info[style]:
            # sampling an artist's works of a specific genre which contains the most works
            # if the num of works is less than train_num+test_num, skip the artist
            if artist == "Unknown Artist":
                continue
            genre_max_num = 0
            for genre in image_info[style][artist]:
                if genre == "Unknown Genre" or genre == "sketch_and_study":
                    continue
                if image_info[style][artist][genre] > genre_max_num:
                    genre_max_num = image_info[style][artist][genre]
                    max_genre = genre
            if genre_max_num < train_num + test_num:
                continue

            image_dir = f"{args.origin_images_dir}/{style}/{artist}/{max_genre}"
            print(f"sampling images from {image_dir}")
            sample_dir = f"{args.exp_data_dir}/sample_resolution_{args.resolution}/{style}/{artist}"
            train_dir, test_dir = f"{sample_dir}/train_clean", f"{sample_dir}/test"
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            images = [file for file in os.listdir(image_dir) if file.endswith(".png")]
            #print(len(images))
            images = sample(images, train_num+test_num)
            train_images, test_images = images[:train_num], images[train_num:]

            # generate training set
            metadata = []
            for image_name in train_images:
                image_path = f"{image_dir}/{image_name}"
                image = center_crop(resize(Image.open(image_path)))
                input = processor(image, return_tensors="pt").to("cuda")
                out = model.generate(**input)
                caption = processor.decode(out[0], skip_special_tokens=True)
                caption = f"{caption} in {artist} style"
                image.save(f"{train_dir}/{image_name}")
                metadata.append({'file_name': image_name, 'text': caption})
            with jsonlines.open(f'{train_dir}/metadata.jsonl', 'w') as writer:
                writer.write_all(metadata)

            # generate test set
            test_text = []
            for image_name in test_images:
                image_path = f"{image_dir}/{image_name}"
                image = Image.open(image_path)
                input = processor(image, return_tensors="pt").to("cuda")
                out = model.generate(**input)
                caption = processor.decode(out[0], skip_special_tokens=True)
                caption = f"{caption} in {artist} style"
                test_text.append(caption)
            with open(f"{test_dir}/text.json", 'w') as file:
                json.dump(test_text, file)


if __name__ == "__main__":
    random.seed(0)    
    parser = argparse.ArgumentParser(description='diffusion attack')
    parser.add_argument('--blip_model', default="/root/autodl-tmp/model/Blip-image-captioning-base", type=str)
    parser.add_argument('--origin_images_dir', default="/root/autodl-tmp/dataset/origin_images", type=str)
    parser.add_argument('--exp_data_dir', default="/root/autodl-tmp/dataset", type=str)
    parser.add_argument('--resolution', default=512, type=str)
    args = parser.parse_args()
    main(args)

