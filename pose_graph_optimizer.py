import numpy as np
from collections import deque

import scipy.linalg
import scipy.optimize
from utils import inverse_transformation, multiply_transformation, twist2HomogMatrix, HomogMatrix2twist
from initialise_vo import Bootstrap
import cv2
import scipy
    
class PoseGraphOptimizer:

    def __init__(self, K, initial_transform, sliding_window_size = 3):
        self.K = K
        self.sliding_window_size = sliding_window_size
        self.images = deque(maxlen=sliding_window_size)
        self.relative_transforms = deque(maxlen=self.sliding_window_size) # X by Y by 12, where relative_transforms[X,Y,:] is T_XY (transform from Y to X)
        # Note: Relative transforms could be size self.sliding_window_size - 1, as we don't need relative transforms to the last tracked frame. But for ease of indexing we store it anyways
        self.transform_to_world = deque(maxlen=self.sliding_window_size) # N x 12


    def add_image(self, new_image, transform_estimate):
        """
        Compute set of relative transforms from the images we currently have to the new image we get
         ### Parameters
            - new_image np.array (image size)
                - A new image frame
            - transform_estimate np.array shape (3, 4)
                - the current estimate T_cw for this new image frame
        """
        self.images.append(new_image)

        transforms = []
        for i in range(len(self.images) - 1):
            image = self.images[i]
            ## Compute transform using DLT from all images in window to new image (reuse sift code lol)

            # Feature extraction
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(image, None)
            kp2, des2 = sift.detectAndCompute(new_image, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                # if m.distance < 0.8*n.distance or n.distance < 0.8*m.distance:
                if m.distance < 0.8*n.distance:
                    good.append([m])

            # Essential matrix by 8p algorithm
            points1 = np.array([kp1[match[0].queryIdx].pt for match in good])
            points2 = np.array([kp2[match[0].trainIdx].pt for match in good])
            E, ransac_inliers = cv2.findEssentialMat(points1, points2, self.K, method=cv2.FM_RANSAC, prob=0.999, threshold=1)
            ransac_inliers = ransac_inliers.astype(np.bool_).reshape(-1)

            _, R, t, _ = cv2.recoverPose(E, points1[ransac_inliers], points2[ransac_inliers], self.K)
            transforms.append(np.row_stack((np.column_stack((R, t)), [0,0,0,1])))

        self.relative_transforms.append(transforms)
        self.transform_to_world.append(HomogMatrix2twist(np.row_stack((transform_estimate, [0,0,0,1]))))

    def optimize(self, with_pattern=True):
        pattern = None
        values_to_optimize = np.array(self.transform_to_world).flatten()
        if with_pattern:
            num_error_terms = int(((len(self.relative_transforms) - 1) * len(self.relative_transforms))/2) # equals sum of range 1 to (len(self.relative_transforms) - 1)

            pattern = scipy.sparse.lil_matrix((num_error_terms, values_to_optimize.shape[0]), dtype=np.int8)
            for i in range(1, len(self.transform_to_world)):
                pattern[:i*6] = 1 # each error is affected by i poses, which are 6 values each

            pattern = scipy.sparse.csr_matrix(pattern)

        # Define the function that computes the residuals
        def residuals(world_to_cameras, poses):
            # world_to_cameras: the parameters to optimize
            # poses: the set of relative camera poses we use to inform the optimisation

            world_to_cameras = world_to_cameras.reshape((-1,6)) # reshape into array of transformation matrices
            residuals = []
            for i in range(1, len(poses)): # loop over the transforms from frame i to j. Start at 1 because there is no residual for the first image
                
                T_iw = twist2HomogMatrix(world_to_cameras[i]) # transform from w to i
                
                for j in range(i): # there should be i relative transformation matrices for the i-th pose we are optimising
                    T_jw = twist2HomogMatrix(world_to_cameras[j])
                    T_ij = poses[i][j] # relative_transform from pose j to pose i

                    residual = scipy.linalg.norm((T_iw - T_jw @ T_ij)[:3,:]) # compute difference between current estimate for world position of camera, and what we get from relative transforms
                    # ^ WTF it should be T_iw - T_ij @ T_jw but it only actually works when i do it this way what did i do wrong???
                    residuals.append(residual)
            return np.array(residuals).flatten()


        result = scipy.optimize.least_squares(residuals, values_to_optimize, args=(self.relative_transforms,), max_nfev=20, jac_sparsity=pattern)
        print("Pose Graph Optimised: ")
        for value in (result.x).reshape(-1,6):
            print(twist2HomogMatrix(value)[:3,:])
        print("Regular output: ") # Separator for when the non-optimised value gets printed by the rest of the code
        return (result.x).reshape(-1,6)

