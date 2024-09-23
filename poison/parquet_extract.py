import os
import json
import argparse
import pandas as pd
from io import BytesIO
from PIL import Image

def main(args):
    parquet_dir = f"{args.dataset_dir}/data"
    info_path = f"{args.dataset_dir}/dataset_infos.json"
    
    with open(info_path, 'r') as file:
        info = json.load(file)
    artist_info = info["huggan--wikiart"]["features"]["artist"]["names"]
    genre_info = info["huggan--wikiart"]["features"]["genre"]["names"]
    style_info = info["huggan--wikiart"]["features"]["style"]["names"]

    # processing parquet files in the dataset
    parquet_counter = 0
    image_counter = {}
    for file in os.listdir(parquet_dir):
        if not file.endswith(".parquet"):
            continue
        parquet_counter += 1
        print(f"extracting images from {file} [{parquet_counter}/72]...")
        df = pd.read_parquet(f"{parquet_dir}/{file}")

        for index, row in df.iterrows():
            image = row["image"]["bytes"]
            artist = artist_info[row['artist']]
            genre = genre_info[row['genre']]
            style = style_info[row['style']]
            # image saving directory {args.export_dir}/{style}/{artist}/{genre}
            # count images in the directory, image name {counter}.jpg
            if style not in image_counter:
                image_counter[style] = {}
            if artist not in image_counter[style]:
                image_counter[style][artist] = {}
            if genre not in image_counter[style][artist]:
                image_counter[style][artist][genre] = 0
            image_counter[style][artist][genre] += 1
            #save_dir = f"{args.export_dir}/{style}/{artist}/{genre}"
            #os.makedirs(save_dir, exist_ok=True)
            #image = Image.open(BytesIO(image))
            #image.save(f"{save_dir}/{image_counter[style][artist][genre]}.jpg")
            
    json_path = f"{args.export_dir}/info.json"
    with open(json_path, 'w') as f:
        json.dump(image_counter, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parquet extracting')
    parser.add_argument('--dataset_dir', default='/root/autodl-tmp/dataset/wikiart', type=str)
    parser.add_argument('--export_dir', default='/root/autodl-tmp/dataset/origin_images', type=str)
    args = parser.parse_args()
    main(args)

