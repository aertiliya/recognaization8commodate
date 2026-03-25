import cv2, json, os, argparse
from pathlib import Path
from tqdm import tqdm

CLASS_NAMES = {'hks_large': 0, 'hks_small': 1, 'hn_can': 2, 'jlb_can': 3, 'kkkl_can': 4, 'wlj_can': 5, 'xb_wt': 6, 'xb': 7}
ROI_SIZE = (224, 224)

def parse_json_label(json_path):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    objects = []
    for shape in data.get('shapes', []):
        label, points = shape.get('label', ''), shape.get('points', [])
        if label not in CLASS_NAMES or len(points) < 2: continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        objects.append((label, [min(xs), min(ys), max(xs), max(ys)]))
    return objects

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    print(f"\n✂️ 开始裁剪: {input_dir} -> {output_dir}")
    print(f"找到 {len(json_files)} 个 JSON 标签")
    
    for cls in CLASS_NAMES.keys(): os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
    stats = {k: 0 for k in CLASS_NAMES.keys()}
    for json_path in tqdm(json_files, desc="裁剪进度"):
        base_name = json_path.stem
        # 匹配对应图片
        img_path = next((input_path / f"{base_name}{ext}" for ext in ['.jpg', '.jpeg', '.png'] if (input_path / f"{base_name}{ext}").exists()), None)
        if not img_path: continue
        
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        for idx, (label, bbox) in enumerate(parse_json_label(str(json_path))):
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            roi = cv2.resize(image[y1:y2, x1:x2], ROI_SIZE, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_dir, label, f"{base_name}_roi{idx}.jpg"), roi)
            stats[label] += 1
            
    print(f"✅ 裁剪完成！总计: {sum(stats.values())} 张商品特写图")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    process_dataset(args.input, args.output)