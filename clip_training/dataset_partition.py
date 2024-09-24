import os
import csv
import random
import argparse
from PIL import Image
from random import shuffle


def main(args):
    train_images_all, valid_images_all, test_images_all = [], [], []
    # travelling all images of a specify style and save infos
    for style in os.listdir(args.image_dir):
        style_dir = f"{args.image_dir}/{style}"
        images = []
        
        for root, dirs, files in os.walk(style_dir):
            for file in files:
                if file.endswith(".jpg"):
                    image_path = f"{root}/{file}"
                    images.append({"image_path": image_path, "style": style})

        # devide images of a specific style into train_images, valid_images and test_images by 8:1:1
        shuffle(images)
        train_num = int(len(images) * 0.8)
        train_images = images[:train_num]
        valid_images = images[train_num:]
        valid_num = int(len(valid_images) * 0.5)
        test_images = valid_images[valid_num:]
        valid_images = valid_images[:valid_num]

        train_images_all.extend(train_images)
        valid_images_all.extend(valid_images)
        test_images_all.extend(test_images)

    # generate training set, validation set, and test set(csv file and images) 
    shuffle(train_images_all)
    shuffle(valid_images_all)
    shuffle(test_images_all)
    os.makedirs(f"{args.export_dir}/train", exist_ok=True)
    for i in range(len(train_images_all)):
        if i % 1000 == 0:
            print(f"generating training set({i+1}/{len(train_images_all)} images)")
        image_path = train_images_all[i]["image_path"]
        image = Image.open(image_path)
        image.save(f"{args.export_dir}/train/{i}.jpg")
    with open(f'{args.export_dir}/train.csv', 'w', newline='') as csvfile:  
        fieldnames = ['image_path', 'style']  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  
        writer.writerows(train_images_all)

    os.makedirs(f"{args.export_dir}/valid", exist_ok=True)
    for i in range(len(valid_images_all)):
        if i % 1000 == 0:
            print(f"generating validation set({i+1}/{len(valid_images_all)} images)")
        image_path = valid_images_all[i]["image_path"]
        image = Image.open(image_path)
        image.save(f"{args.export_dir}/valid/{i}.jpg")
    with open(f'{args.export_dir}/valid.csv', 'w', newline='') as csvfile:  
        fieldnames = ['image_path', 'style']  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  
        writer.writerows(valid_images_all)

    os.makedirs(f"{args.export_dir}/test", exist_ok=True)
    for i in range(len(test_images_all)):
        if i % 1000 == 0:
            print(f"generating test set({i+1}/{len(test_images_all)} images)")
        image_path = test_images_all[i]["image_path"]
        image = Image.open(image_path)
        image.save(f"{args.export_dir}/test/{i}.jpg")
    with open(f'{args.export_dir}/test.csv', 'w', newline='') as csvfile:  
        fieldnames = ['image_path', 'style']  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  
        writer.writerows(test_images_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parquet extracting')
    parser.add_argument('--image_dir', default='/root/autodl-tmp/dataset/origin_images', type=str)
    parser.add_argument('--export_dir', default='/root/autodl-tmp/dataset/clip_training', type=str)
    args = parser.parse_args()

    random.seed(0)
    main(args)

