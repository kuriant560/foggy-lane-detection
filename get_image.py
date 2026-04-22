import json
import shutil
import os

labels_path = 'bdd100k_data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
images_dir = 'bdd100k_data/bdd100k/bdd100k/images/100k/train'

try:
    with open(labels_path, 'r') as f:
        data = json.load(f)

    found = False
    for item in data:
        if item['attributes']['weather'] == 'foggy' and item['attributes']['scene'] == 'highway':
            img_name = item['name']
            src = os.path.join(images_dir, img_name)
            if os.path.exists(src):
                shutil.copy(src, 'test_foggy_highway.jpg')
                print(f"Success! Copied {img_name} to test_foggy_highway.jpg")
                found = True
                break

    if not found:
        print("Could not find a foggy highway image. Falling back to a standard foggy image.")
        for item in data:
            if item['attributes']['weather'] == 'foggy':
                img_name = item['name']
                src = os.path.join(images_dir, img_name)
                if os.path.exists(src):
                    shutil.copy(src, 'test_foggy_highway.jpg')
                    print(f"Success! Copied {img_name} to test_foggy_highway.jpg")
                    break
except Exception as e:
    print("Error:", e)
