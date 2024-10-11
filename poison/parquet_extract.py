import os
import json
import argparse
import threading
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

image_counter = {}
lock = threading.Lock()

def process_parquet(args, file, artist_info, genre_info, style_info):
    parquet_path = f"{args.dataset_dir}/data/{file}"
    df = pd.read_parquet(parquet_path)
    for index, row in df.iterrows():
        image = row["image"]["bytes"]
        artist = artist_info[row['artist']]
        genre = genre_info[row['genre']]
        style = style_info[row['style']]
        # image saving directory {args.export_dir}/{style}/{artist}/{genre}
        # count images in the directory, image name {counter}.jpg
        with lock:
            if style not in image_counter:
                image_counter[style] = {}
            if artist not in image_counter[style]:
                image_counter[style][artist] = {}
            if genre not in image_counter[style][artist]:
                image_counter[style][artist][genre] = 0
            image_counter[style][artist][genre] += 1
        save_dir = f"{args.export_dir}/{style}/{artist}/{genre}"
        os.makedirs(save_dir, exist_ok=True)
        image = Image.open(BytesIO(image))
        image.save(f"{save_dir}/{image_counter[style][artist][genre]}.png")

    return file

def main(args):
    parquet_dir = f"{args.dataset_dir}/data"
    info_path = f"{args.dataset_dir}/dataset_infos.json"
    
    with open(info_path, 'r') as file:
        info = json.load(file)
    artist_info = info["huggan--wikiart"]["features"]["artist"]["names"]
    genre_info = info["huggan--wikiart"]["features"]["genre"]["names"]
    style_info = info["huggan--wikiart"]["features"]["style"]["names"]

    # using thread pool to process parquet files in the dataset
    with ThreadPoolExecutor(max_workers=args.thread_pool_worker) as t:
        task_list = []
        for file in os.listdir(parquet_dir):
            if not file.endswith(".parquet"):
                continue
            task = t.submit(process_parquet, args, file, artist_info, genre_info, style_info)
            print(f"extracting images from {file}")
            task_list.append(task)

        parquet_counter = 0
        for future in as_completed(task_list):
            data = future.result()
            parquet_counter += 1
            print(f"finished extracting images from {data}[{parquet_counter}/72]")
            
    json_path = f"{args.export_dir}/info.json"
    with open(json_path, 'w') as f:
        json.dump(image_counter, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parquet extracting')
    parser.add_argument('--dataset_dir', default='/root/autodl-tmp/dataset/wikiart', type=str)
    parser.add_argument('--export_dir', default='/root/autodl-tmp/dataset/origin_images', type=str)
    parser.add_argument('--thread_pool_worker', default=72, type=int)
    args = parser.parse_args()
    main(args)

