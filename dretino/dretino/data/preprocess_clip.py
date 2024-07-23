# Resizing the testing images
import os
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percetage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """
    percentage = 0.02

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(im_crop)


def save_single(args):
    img_file, input_path_folder, output_path_folder, output_size = args
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_original)

    image.save(os.path.join(output_path_folder + img_file))


def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    """
    Uses multiprocessing to make it fast
    """
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))
