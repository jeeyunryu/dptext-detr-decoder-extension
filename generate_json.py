import os
import json
import cv2
# from tqdm import tqdm 
from scipy.interpolate import splprep, splev
import numpy as np
import string
import re # regular expression. 문자열 안에서 숫자를 추출하기 위함.

def interpolate_polygon(coords, num_points=16): # 상하 변만 보간
    x = coords[0::2] + [coords[0]]
    y = coords[1::2] + [coords[1]]

    new_x = []
    new_y = []

    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]

        for t in np.linspace(0, 1, num_points // 4, endpoint=False):
            new_x.append(x1 + t * (x2 - x1))
            new_y.append(y1 + t * (y2 - y1))

    coords_interp = []
    for xi, yi in zip(new_x, new_y):
        coords_interp.extend([xi, yi])
    
    return coords_interp



def polygon_to_bbox(polygon):
    x_coords = polygon[0::2] # polygon[start:stop:step] start를 기준으로 짝수번째 원소만을 반환하겠다는 뜻 x 값에 해당하는 값들만을 반환함
    y_coords = polygon[1::2]
    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def process_dataset(image_dir, annotation_dir, output_json):
    images = []
    annotations = []
    categories = [{
        "supercategory": "beverage",
        "id": 1,
        "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
        "name": "text"
    }]

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))], key=lambda x: int(re.search(r'\d+', x).group()))
    annotation_id = 1
    image_id = 0

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        images.append({
            "width": width,
            "height": height,
            "file_name": image_file,
            "id": image_id,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        file_name = os.path.splitext(image_file)[0]
        file_num = re.search(r'\d+', file_name).group().zfill(7) #제로패딩

        # base_name = os.path.splitext(image_file)[0].replace("MPSC_", "") # 파일명을 이름과 확장자로 나누는 함수
        # ann_file = f"gt_{base_name}.txt"
        ann_file = f"{file_num}.txt"
        ann_path = os.path.join(annotation_dir, ann_file)

        if not os.path.exists(ann_path):
            print(f"Warning: annotation {ann_path} not found.")
            image_id += 1
            continue

        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(',') # strip은 양쪽의 공백 문자 제거 띄어쓰기, 개행 문자, 탭 모두 포함
            if len(parts) < 9 or '' in parts[:8]: 
                continue
            coords = list(map(float, parts[:8]))  # x1,y1,x2,y2,x3,y3,x4,y4
            coords = interpolate_polygon(coords, num_points=16) 

            text = parts[8] if len(parts) > 8 else ""
            # segmentation = coords
            bbox = polygon_to_bbox(coords)
            upper_text = text.upper()
            words = list(text)
            rec = [96] * 25


            

            for i, word in enumerate(words):
                if i < 25:  # rec 리스트의 길이를 넘지 않도록
                    c = word
                    if c in string.printable and 32 <= ord(c) <= 126:
                        rec[i] = ord(c) - 32  # ' ' (space) 기준 0~94
                    else:
                        rec[i] = 0



            annotations.append({
                "image_id": image_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "rec": rec,
                "category_id": 1,
                "iscrowd": 0,
                "id": annotation_id,
                "polys": coords,
                "iou": 1.0,
            })
            annotation_id += 1

        image_id += 1

    coco_output = {
        "licenses": [],
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w') as out_file:
        json.dump(coco_output, out_file, indent=4)
    print(f"✅ COCO JSON saved to: {output_json}")

# 사용 예
image_dir = "datasets/MPSC/test_images"
annotation_dir = "datasets/MPSC/annotation/test"
output_json = "datasets/MPSC/mpsc_test_coco.json"

process_dataset(image_dir, annotation_dir, output_json)
