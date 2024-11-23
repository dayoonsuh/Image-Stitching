"""Implements image stitching."""

from PIL import Image
import cv2
import numpy as np
import skimage
import torch
import torch.nn.functional as F
import kornia
from helpers import plot_inlier_matches, compute_harris_response, get_harris_points, plot_harris_points
import matplotlib.pyplot as plt

from skimage.transform import ProjectiveTransform, warp


class ImageStitcher(object):
    def __init__(self, img1, img2, keypoint_type='harris', descriptor_type='pixel'):
        """
        Inputs:
            img1: h x w tensor.
            img2: h x w tensor.
            keypoint_type: string in ['harris']
            descriptor_type: string in ['pixel', 'hynet']
        """
        self.img1 = img1.squeeze()
        self.img2 = img2.squeeze()
        self.keypoint_type = keypoint_type
        self.descriptor_type = descriptor_type
        #### Your Implementation Below ####
        self.threshold = 1000
        self.num_iter = 10000
        self.patch_size = 11

        print(f"Type: {self.descriptor_type}, N: {self.num_iter}, Threshold: {self.threshold}, Patch size: {self.patch_size}")

        # Extract keypoints
        self.keypoints1 = self._get_keypoints(kornia.utils.tensor_to_image(self.img1))  # length 540 # type is ndarray [(a,b)..]
        self.keypoints2 = self._get_keypoints(
            kornia.utils.tensor_to_image(self.img2))  

        # Extract descriptors at each keypoint
        self.desc1 = self._get_descriptors(self.img1, self.keypoints1) 
        self.desc2 = self._get_descriptors(self.img2, self.keypoints2)

        # Compute putative matches and match the keypoints
        matches = self._get_putative_matches(self.desc1, self.desc2, 300)  # [2, num_max_matches]
        matched_keypoints = torch.empty((matches.shape[1], 4))

        for idx in range(matches.shape[1]):
            kp1_idx = int(matches[0][idx])
            kp2_idx = int(matches[1][idx])
            matched_keypoints[idx] = torch.tensor(
                self.keypoints1[kp1_idx][::-1] + self.keypoints2[kp2_idx][::-1])

        # Perform RANSAC to find the best homography and inliers
        inliers, best_homography = self._ransac(matched_keypoints, self.num_iter, self.threshold)
        # Plot the inliers
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_inlier_matches(ax,
                            kornia.utils.tensor_to_image(img1),
                            kornia.utils.tensor_to_image(img2),
                            matched_keypoints[inliers])
        plt.tight_layout()
        plt.savefig('inlier_matches_%s.png' % self.descriptor_type)

        # Refit with all inliers
        matched_keypoints = matched_keypoints[inliers]
        best_inliers, final_homography = self._ransac(
            matched_keypoints, num_iterations=int(self.num_iter/10), inlier_threshold=50)
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_inlier_matches(ax,
                            kornia.utils.tensor_to_image(img1),
                            kornia.utils.tensor_to_image(img2),
                            matched_keypoints[best_inliers])
        plt.tight_layout()
        plt.savefig('refit_inlier_matches_%s.png' % self.descriptor_type)

        stitched = self.stitch(final_homography)
        print("Done stitching")

        plt.figure()
        plt.imshow(stitched)
        plt.gray()
        plt.tight_layout()
        plt.savefig('stitched_%s.png' % self.descriptor_type)

    def _get_keypoints(self, img):
        """
        Extract keypoints from the image.

        Inputs:
            img: h x w tensor.
        Outputs:
            keypoints: N x 2 numpy array.
        """
        harrisim = compute_harris_response(img)
        keypoints = get_harris_points(harrisim)
        return keypoints

    def _get_descriptors(self, img, keypoints):
        """
        Extract descriptors from the image at the given keypoints.

        Inputs:
            img: h x w tensor.
            keypoints: N x 2 tensor.
        Outputs:
            descriptors: N x D tensor.
        """
        if self.descriptor_type == 'pixel':
            patch_size = self.patch_size
            pad = patch_size//2
            padding = (pad, pad, pad, pad)
            padded = torch.nn.functional.pad(img, padding, mode='constant', value=0)
            patches = padded.unfold(0, patch_size, 1).unfold(1, patch_size, 1)

            descriptors = torch.empty((len(keypoints), patch_size**2))

            for i in range(len(keypoints)):
                x, y = keypoints[i]
                descriptor = patches[x, y].flatten()
                descriptor_centralized = descriptor - torch.mean(descriptor)
                descriptor_normalized = descriptor_centralized / torch.norm(descriptor_centralized, p=2)
                descriptors[i] = descriptor_normalized.unsqueeze(0)

        elif self.descriptor_type == 'hynet':
            hynet = kornia.feature.HyNet(pretrained=True)
            patch_size = 32
            pad = patch_size//2
            padding = (pad, pad-1, pad, pad-1)
            padded = torch.nn.functional.pad(img, padding, mode='constant', value=0)
            patches = padded.unfold(0, patch_size, 1).unfold(1, patch_size, 1)

            descriptors = torch.empty((len(keypoints), patch_size**2))

            for i in range(len(keypoints)):
                x, y = keypoints[i]
                descriptor = patches[x, y].flatten()
                descriptor_centralized = descriptor - torch.mean(descriptor)
                descriptor_normalized = descriptor_centralized / torch.norm(descriptor_centralized, p=2)
                descriptors[i] = descriptor_normalized.unsqueeze(0)

            descriptors = descriptors.reshape(len(keypoints), 1, 32, 32)
            descriptors = hynet(descriptors)

        return descriptors

    def _get_putative_matches(self, desc1, desc2, max_num_matches=100):
        """
        Compute putative matches between two sets of descriptors.

        Inputs:
            desc1: N x D tensor.
            desc2: M x D tensor.
            max_num_matches: Integer
        Outputs:
            matches: 2 x max_num_matches tensor.
        """
        distances = torch.cdist(desc1, desc2, p=2).flatten()
        sorted_idx = torch.argsort(distances, dim=0)[
            :max_num_matches]  # [N * M]
        desc1_idx, desc2_idx = torch.unravel_index(
            sorted_idx, (desc1.shape[0], desc2.shape[0]))

        matches = torch.stack((desc1_idx, desc2_idx), dim=0)

        return matches

    def _get_homography(self, matched_keypoints):
        """
        Compute the homography between two images.

        Inputs:
            matched_keypoints: N x 4 tensor.
        Outputs:
            homography: 3 x 3 tensor.
        """
        A = []
        for keypoint in matched_keypoints:
            x1, y1, x2, y2 = keypoint
            A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2])
            A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2, -y2])

        A = torch.tensor(A, dtype=torch.float32)
        U, S, V = torch.linalg.svd(A)

        h = V[-1, :].view(3, 3)
        homography = h/h[2, 2]

        return homography

    def _homography_inliers(self, H, matched_keypoints, inlier_threshold):
        """
        Compute the inliers for the given homography.

        Inputs:
            H: Homography 3 x 3 tensor.
            matched_keypoints: N x 4 tensor.
            inlier_threshold: upper bounds on what counts as inlier.
        Outputs:
            inliers: N tensor, indicates whether each matched keypoint is an inlier.
        """
        N = matched_keypoints.shape[0]
        inliers = torch.zeros(N)

        keypoints1 = matched_keypoints[:, :2]  
        keypoints2 = matched_keypoints[:, 2:] 

        ones = torch.ones((keypoints1.shape[0], 1))
        keypoints1 = torch.cat((keypoints1, ones), dim=1)
        transformed = torch.matmul(H, keypoints1.T) 
        w = transformed[2, :]
        transformed_keypoints1 = transformed / w  
        residuals = torch.norm(transformed_keypoints1.T[:, :2] - keypoints2, dim=1)**2
        inliers = residuals < inlier_threshold
        return inliers

    def _ransac(self, matched_keypoints, num_iterations, inlier_threshold):
        """
        Perform RANSAC to find the best homography.

        Inputs:
          matched_keypoints: N x 4 tensor.
          num_iterations: Number of iteration to run RANSAC.
          inlier_threshold: upper bounds on what counts as inlier.
        Outputs:
          best_inliers: N tensor, indicates whether each matched keypoint is an inlier.
          best_homography: 3 x 3 tensor
        """
        max_inliers = 0
        for i in range(num_iterations):
            random_indices = torch.randperm(len(matched_keypoints))[:4]
            # [4, 4] four pairs
            random_points = matched_keypoints[random_indices]
            homography = self._get_homography(random_points)

            homography_inliers = self._homography_inliers(
                homography, matched_keypoints, inlier_threshold)
            num_inliers = torch.sum(homography_inliers)

            if num_inliers > max_inliers:
                best_inliers = homography_inliers
                best_homography = homography.clone()
                max_inliers = num_inliers

        # compute mean residual
        inliers_keypoints = matched_keypoints[best_inliers]
        keypoints1 = inliers_keypoints[:, :2]  
        keypoints2 = inliers_keypoints[:, 2:] 
        ones = torch.ones((keypoints1.shape[0], 1))
        keypoints1 = torch.cat((keypoints1, ones), dim=1)
        transformed = torch.matmul(homography, keypoints1.T) 
        w = transformed[2, :]
        transformed_keypoints1 = transformed / w  # [3, num-max match]
        residuals = torch.norm(
            transformed_keypoints1.T[:, :2] - keypoints2, dim=1)**2
        mean_residuals = torch.mean(residuals)

        print(f"Number of inliers: {max_inliers}, Mean residuals: {mean_residuals}")

        return best_inliers, best_homography

    def stitch(self, final_homography):
        """
        Stitch the two images together.

        Inputs:
            final_homography: 3 x 3 tensor.
        Outputs:
            stitched: h x w tensor.
        """
        img1 = self.img1.numpy()
        img2 = self.img2.numpy()
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        transform = ProjectiveTransform(final_homography)
        corners_img1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]])
        corners_img2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]])
        warped_corners_img1 = transform(corners_img1)
        all_corners = np.vstack((warped_corners_img1, corners_img2))
        min_corner = np.min(all_corners, axis=0)
        max_corner = np.max(all_corners, axis=0)
        output_shape = np.ceil(max_corner - min_corner).astype(int)[::-1]

        translation_matrix = np.array([[1, 0, -min_corner[0]],
                                       [0, 1, -min_corner[1]],
                                       [0, 0, 1]])
        translation = ProjectiveTransform(translation_matrix)

        warped_img1 = warp(img1, (transform + translation),
                           output_shape=output_shape, cval=-1)
        warped_img2 = warp(img2, translation.inverse,
                           output_shape=output_shape, cval=-1)
        
        blank_img1 = warp(img1, (transform + translation).inverse,
                          output_shape=output_shape, cval=0)
        blank_img2 = warp(img2, translation.inverse,
                          output_shape=output_shape, cval=0)
             
        mask_img1 = (warped_img1 != -1).astype(int)
        mask_img2 = (warped_img2 != -1).astype(int)

        overlap = mask_img1 + mask_img2

        overlap += (overlap == 0).astype(int)
        stitched = (blank_img1+blank_img2)/overlap

        stitched = np.where(blank_img1 > 0, blank_img1, stitched)
     
        return stitched
