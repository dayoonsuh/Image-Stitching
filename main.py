"""Implements the main file for part1."""

import argparse
import kornia
import numpy as np
import torch
from PIL import Image

from image_stitcher import ImageStitcher
import kornia
from helpers import plot_inlier_matches, compute_harris_response, get_harris_points
# Fixed seed
torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser(
        description='CS59300CVD Assignment 3 Part 1')
    parser.add_argument('-k', '--keypoint_type', default='harris',
                        type=str, help="Type of keypoint detector.")
    parser.add_argument('-d', '--descriptor_type', default='hynet',
                        type=str, help="Type of descriptor; Supports ['pixel', 'hynet'].")
    args = parser.parse_args()

    img1_path = 'data/uttower_left.jpg'
    img2_path = 'data/uttower_right.jpg'

    # Load image to tensor
    img1 = Image.open(img1_path)
    img1 = kornia.utils.image_to_tensor(np.array(img1), False).float() / 255.
    img1 = kornia.color.rgb_to_grayscale(img1)
    img2 = Image.open(img2_path)
    img2 = kornia.utils.image_to_tensor(np.array(img2), False).float() / 255.
    img2 = kornia.color.rgb_to_grayscale(img2)  # type: torch.Tensor

    # Stitch images
    ImageStitcher(img1, img2, keypoint_type=args.keypoint_type,
                  descriptor_type=args.descriptor_type)


if __name__ == '__main__':
    main()
