import rawpy
import imageio
import os
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor


parser = ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='preprocess.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

raw_folder = config['raw_folder']
out_folder = config['out_folder']

os.makedirs(out_folder, exist_ok=True)

for filename in tqdm(os.listdir(raw_folder)):
    if filename.lower().endswith('.arw'):
        out_filename = filename.replace('.ARW', '.png').replace('.arw', '.png')
        out_path = os.path.join(out_folder, out_filename)

        # Skip if the output .png file already exists
        if os.path.exists(out_path):
            continue

        path = os.path.join(raw_folder, filename)
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        imageio.imwrite(out_path, rgb)
def validate_images(folder):
    validated = True
    for root, _, files in os.walk(folder):
        for fname in tqdm(files):
            if fname.lower().endswith('.png'):
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        img = pil_to_tensor(img)
                        if torch.isnan(img).any() or torch.isinf(img).any():
                            raise Exception
                except Exception as e:
                    validated = False
                    print(f"Corrupted: {path} ({e})")
                    os.remove(path)
                    # Redo conversion
                    with rawpy.imread(path) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
                    out_path = os.path.join(out_folder, fname.replace('.arw', '.png').replace('.ARW', '.png'))
                    imageio.imwrite(out_path, rgb)
                    if not validated:
                        validate_images(folder)

# Call this once
validate_images(out_folder)