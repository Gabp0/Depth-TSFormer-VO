from transformers import pipeline
from PIL import Image
from os import listdir, makedirs, path
from tqdm import tqdm
import numpy as np

SEQUENCES_PATH = '/home/gopontarolo/siamese-vo/Datasets/kitti-odometry/dataset/sequences'
SEQUENCES = ['02', '03', '04', '05', '06', '07', '08', '09', '10']

depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

mean_depths = []
std_depths = []
for seq in SEQUENCES:
    img_dir = path.join(SEQUENCES_PATH, seq, 'image_0')
    depth_dir = path.join(SEQUENCES_PATH, seq, 'depth_0')
    if not path.exists(depth_dir):
        makedirs(depth_dir)
    
    img_files = sorted(listdir(img_dir))
    for img_file in tqdm(img_files):
        img_path = path.join(img_dir, img_file)
        depth_path = path.join(depth_dir, img_file)

        image = Image.open(img_path)
        depth = depth_pipe(image)['depth']
        depth.save(depth_path)

        # calculate mean and stddev of depth map
        depth_np = np.array(depth)
        mean_depth = np.mean(depth_np)
        std_depth = np.std(depth_np)
        mean_depths.append(mean_depth)
        std_depths.append(std_depth)
    print(f"Current Mean Depth: {np.mean(mean_depths)}, Current Std Dev Depth: {np.mean(std_depths)}")

mean_depth_overall = np.mean(mean_depths)
std_depth_overall = np.mean(std_depths)
print(f"Overall Mean Depth: {mean_depth_overall}, Overall Std Dev Depth: {std_depth_overall}")