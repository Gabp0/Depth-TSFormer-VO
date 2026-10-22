from PIL import Image
from os import listdir, makedirs, path
from tqdm import tqdm
import numpy as np
from block_matching import block_matching

SEQUENCES_PATH = 'Data/Datasets/kitti-odometry/dataset/sequences'
SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

for seq in SEQUENCES:
    img_dir = path.join(SEQUENCES_PATH, seq, 'image_0')
    img2_dir = path.join(SEQUENCES_PATH, seq, 'image_1')
    disp_dir = path.join(SEQUENCES_PATH, seq, 'disp_0')
    if not path.exists(disp_dir):
        makedirs(disp_dir)
    
    img_files = sorted(listdir(img_dir))
    for img_file in tqdm(img_files):
        img_path = path.join(img_dir, img_file)
        img2_path = path.join(img2_dir, img_file)
        disp_path = path.join(disp_dir, img_file)

        image = Image.open(img_path)
        image2 = Image.open(img2_path)
        disparity_map = block_matching(np.array(image), np.array(image2), block_size=16, search_range=10)
        disp_image = Image.fromarray((disparity_map / np.max(disparity_map) * 255).astype(np.uint8))
        disp_image.save(disp_path)
