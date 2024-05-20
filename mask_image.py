import os
import csv
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# multimodal model file path
model_path = './model/models--vit-gpt2-image-captioning'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)
feature_extractor = ViTImageProcessor.from_pretrained(model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def add_black_block_random(image, block_size, color):
    h, w, _ = image.shape
    block_x = np.random.randint(0, w - block_size)
    block_y = np.random.randint(0, h - block_size)
    image[block_y:block_y + block_size, block_x:block_x + block_size, :] = color
    return image, block_x, block_y

def add_black_block_repeat(image, block_size, color, block_x, block_y):
    image[block_y:block_y + block_size, block_x:block_x + block_size, :] = color
    return image

def calculate_coverage(image, color):
    # 计算黑色区域的覆盖率
    black_pixels = np.sum(image == color)
    total_pixels = image.size
    coverage_percentage = (black_pixels / total_pixels) * 100
    return coverage_percentage

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# image dataset path
root = "./image_dataset"
# output_image_path = root + '/temp.png' os.listdir(root)

for r in ['pokemon', 'coco2017', 'flickr8k', 'Impressions', 'midjourney-threads', 'laion-coco-aesthetic']: #folder name in imagedataset
    dataset_folder = os.path.join(root, r)
    
    for img_file in os.listdir(dataset_folder):
        if '.jpg' not in img_file and '.png' not in img_file:
            continue
        if img_file.split('.')[0] not in os.listdir(dataset_folder):
            os.mkdir(os.path.join(dataset_folder, img_file.split('.')[0]))
        output_folder = os.path.join(dataset_folder, img_file.split('.')[0])
        # if 'caption.csv' in os.listdir(output_folder):
        #     continue
        block = []
        ratio = []
        col = []
        text = []
        img_path = os.path.join(dataset_folder, img_file)
    
        output = predict_step([img_path])[0]
        block.append(0)
        ratio.append(0)
        col.append(0)
        text.append(output)
        print(r, img_file, output)

        totalx = []
        totaly =[]
        for block_size in [4]:    
            tempx = []
            tempy = []
            for mask_ratio in range(5, 10, 5):
                # black
                for color in [0]:
                    print(r, img_file, block_size)
                    image = cv2.imread(img_path)
                    h, w, _ = image.shape
                    image_size = (h, w, 3)
                    white_image = np.ones(image_size, dtype=np.uint8) * 200  
                    while True:
                        image, x, y = add_black_block_random(image, block_size, color)
                        tempx.append(x)
                        tempy.append(y)
                        white_image = add_black_block_repeat(white_image, block_size, color,x, y)
                        coverage = calculate_coverage(white_image, color)
                        # print(f"Coverage: {coverage:.2f}%")

                        if coverage >= mask_ratio:
                            break
                    output_image_path =  os.path.join(output_folder, str(block_size) + '_' + str(int(round(coverage*100, 2))) + '_' + str(color) + '.png')
                    cv2.imwrite(output_image_path, image)
                    output = predict_step([output_image_path])[0]
                    block.append(block_size)
                    ratio.append(coverage)
                    col.append(color)
                    text.append(output)
                    print(r, img_file, block_size, coverage, color, output)
            totalx.append(tempx)
            totaly.append(tempy)

        for block_size in [4]:    
            for mask_ratio in range(5, 10, 5):
                # gray and white
                for color in [127, 255]:
                    print(r, img_file, block_size)
                    image = cv2.imread(img_path)
                    h, w, _ = image.shape
                    image_size = (h, w, 3)
                    white_image = np.ones(image_size, dtype=np.uint8) * 200  
                    for idx in range(len(totalx[[4].index(block_size)])):
                        image = add_black_block_repeat(image, block_size, color, totalx[[4].index(block_size)][idx], totaly[[4].index(block_size)][idx])
                        white_image = add_black_block_repeat(white_image, block_size, color, totalx[[4].index(block_size)][idx], totaly[[4].index(block_size)][idx])

      
                    output_image_path =  os.path.join(output_folder, str(block_size) + '_' + str(int(round(coverage*100, 2))) + '_' + str(color) + '.png')
                    cv2.imwrite(output_image_path, image)
                    output = predict_step([output_image_path])[0]
                    block.append(block_size)
                    ratio.append(coverage)
                    col.append(color)
                    text.append(output)
                    print(r, img_file, block_size, coverage, color, output)

        # 列表合并成一个二维列表
        data = list(zip(block, ratio, col, text))
        # CSV 文件名
        csv_file = os.path.join(output_folder, 'caption.csv')
        # 写入 CSV 文件
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # 写入表头
            writer.writerow(['block_length', 'mask_ratio', 'color', 'text'])
            
            # 写入数据
            writer.writerows(data)

        print(f"CSV 文件 {csv_file} 写入成功")
