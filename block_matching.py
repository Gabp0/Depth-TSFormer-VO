import cv2
import numpy as np

def block_matching(img1, img2, block_size, search_range):
    height, width = img1.shape
    disparity_map = cv2.Mat.zeros((height, width), dtype=np.float32)

    for y in range(0, height - block_size):
        for x in range(0, width - block_size):
            block1 = img1[y:y + block_size, x:x + block_size]
            best_match = None
            min_ssd = float('inf')

            for j in range(max(0, y - search_range), min(height - block_size, y + search_range)):
                for i in range(max(0, x - search_range), min(width - block_size, x + search_range)):
                    block2 = img2[j:j + block_size, i:i + block_size]
                    ssd = cv2.norm(block1, block2, cv2.NORM_L2)

                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_match = (i, j)

            if best_match is not None:
                disparity_map[y:y + block_size, x:x + block_size] = abs(x - best_match[0])

    return disparity_map