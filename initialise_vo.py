import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *
 
import matplotlib
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class DataLoader():

    def __init__(self, dataset: str):
        """
        Retrieves the images and camera intrinsics for easy switching between
        different datasets.

        ### Parameters
        1. dataset : str
            - Either "kitti", "malaga", or "parking" 
        """
        
        assert dataset in ("kitti", "malaga", "parking")
        
        if dataset == "kitti":
            base_path = Path.cwd().joinpath("datasets", "kitti")
            
            # Load camera parameters for Kitti
            path_calib_data = base_path.joinpath("05", "calib.txt")
            calib_data = np.loadtxt(path_calib_data, dtype='str')
            # slice to get projection matrix for camera 0
            calib_data = calib_data[0,1:] 
            P1 = np.array(calib_data.astype(np.float32)).reshape((3, 4))
            K, R1, T1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
            t1 = T1 / T1[3]
            self.K = K

            # Path to the directory of the images; use image_0 by default
            self.im_dir = base_path.joinpath("05", "image_0")

            # List of paths to all the images; use image_0 by default
            self.all_im_paths = list(self.im_dir.glob("*"))

            # Number of images in the dataset
            self.n_im = len(self.all_im_paths)

        elif dataset in ("malaga", "parking"):
            message = "TODO: implement data loader for malaga and parking"
            raise NotImplementedError(message)


class Bootstrap():

    def __init__(self, data_loader: DataLoader, init_frames: tuple = (0, 2),
                 outlier_tolerance=(None, None, 15)):
        """
        Get an initial point cloud

        ### Parameters
        1. data_loader : DataLoader
            - DataLoader instance initialised with one of the three datasets
        2. init_frames : Tuple[int, int], (default (0, 2))
            - Tuple of two integers that specify what images are used to
              initialise the point cloud. The integers refer to the ordering
              of the files in the directory.
        3. outlier_tolerance : tuple(x_tol: float, y_tol: float, z_tol: float)
            - For overwriting the tolerance values on utils.median_outliers()
              during the get_points() method. Pass None instead of a float 
              value to allow for any value
        """
        
        self.data_loader = data_loader
        self.all_im_paths = data_loader.all_im_paths
        self.init_frames = init_frames
        self.K = data_loader.K
        self.outlier_tolerance = outlier_tolerance
        
        # Filled in by get_points()
        self.E = None
        self.R = None
        self.t = None 
        self.triangulated_points = None
        self.keypoints = None
        self.last_img = None

        # The colour map used to represent depth
        self.c_map_name = "nipy_spectral"
        

    def get_points(self) -> tuple:
        """
        Get keypoints and the corresponding landmarks 

        ### Returns
        Tuple[np.array, np.array]
        A tuple of two numpy arrays: the initial keypoints P0 (from the 
        second initial image) and the corresponding landmarks X0
        """
        # Load images
        path_im1 = self.all_im_paths[self.init_frames[0]].__str__()   
        path_im2 = self.all_im_paths[self.init_frames[1]].__str__()
        im1 = cv2.imread(path_im1, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(path_im2, cv2.IMREAD_GRAYSCALE)

        # Feature extraction
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(im1, None)
        kp2, des2 = sift.detectAndCompute(im2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance or n.distance < 0.8*m.distance:
                good.append([m])

        # Essential matrix by 8p algorithm
        points1 = np.array([kp1[match[0].queryIdx].pt for match in good])
        points2 = np.array([kp2[match[0].trainIdx].pt for match in good])
        self.E, ransac_inliers = cv2.findEssentialMat(points1, points2, self.K, cv2.FM_RANSAC, 0.99, 2)
        ransac_inliers = ransac_inliers.astype(np.bool_).reshape(-1)
        self.keypoints = points2[ransac_inliers]

        _, self.R, self.t, _ = cv2.recoverPose(self.E, points1[ransac_inliers], points2[ransac_inliers], self.K)

        projectionMat1 = self.K @ np.column_stack((np.identity(3), np.zeros(3)))
        projectionMat2 = self.K @ np.column_stack((self.R, self.t))

        self.triangulated_points = cv2.triangulatePoints(projectionMat1, projectionMat2, 
                                                         points1[ransac_inliers].T, points2[ransac_inliers].T)
        self.triangulated_points = self.triangulated_points / self.triangulated_points[3]
        self.triangulated_points = self.triangulated_points[:3, :].T

        # Outlier removal: checking if the points are inside the FOV
        h, w = im1.shape
        alpha = self.K[0, 0]
        in_FOV = check_inside_FOV(alpha, w, h, self.triangulated_points)
        self.triangulated_points = self.triangulated_points[in_FOV]
        self.keypoints = self.keypoints[in_FOV]

        # Outlier removal: remove points whose z-value greatly deviates from the median
        z_mask = median_outliers(self.triangulated_points, *self.outlier_tolerance)
        self.triangulated_points = self.triangulated_points[z_mask]
        self.keypoints = self.keypoints[z_mask]

        return self.keypoints.copy(), self.triangulated_points.copy()

    def draw_landmarks(self, aspect_x=20, aspect_y=10, aspect_z=15):
        """
        Draw the 3D landmarks estimated by get_points()
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=-90, azim=0, roll=-90)
        c_map = plt.get_cmap(self.c_map_name)
        z_range = self.triangulated_points[:, 2].max() - self.triangulated_points[:, 2].min()
        ax.scatter(self.triangulated_points[:, 0], self.triangulated_points[:, 1], 
                   self.triangulated_points[:, 2], marker='o', s=5, 
                   c=self.triangulated_points[:, 2], alpha=0.5, cmap=c_map)
        ax.set_box_aspect((aspect_x, aspect_y, aspect_z))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show(block=True)

    def draw_keypoints(self):
        """
        Draw the 2D keypoints estimated by get_points()
        """

        fig, ax = plt.subplots()
        im_path = self.all_im_paths[self.init_frames[-1]]
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(im, cmap="grey")

        c_map = plt.get_cmap(self.c_map_name)
        sc = ax.scatter(self.keypoints[:, 0], self.keypoints[:, 1], s=4, 
                        c=self.triangulated_points[:, 2], cmap=c_map, alpha=0.5)
        cbar = fig.colorbar(sc, orientation="horizontal")
        cbar.set_label("Distance from camera, SFM units")
        plt.show(block=True)


    def draw_all(self, aspect_x=20, aspect_y=10, aspect_z=15):
        """
        Draws both the landmarks and keypoints
        """
        # LANDMARKS
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(122, projection='3d')
        ax.view_init(elev=-90, azim=0, roll=-90)
        c_map = plt.get_cmap(self.c_map_name)
        z_range = self.triangulated_points[:, 2].max() - self.triangulated_points[:, 2].min()
        ax.scatter(self.triangulated_points[:, 0], self.triangulated_points[:, 1], 
                   self.triangulated_points[:, 2], marker='o', s=5, 
                   c=self.triangulated_points[:, 2], alpha=0.5, cmap=c_map)
        ax.set_box_aspect((aspect_x, aspect_y, aspect_z))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # KEYPOINTS
        ax = fig.add_subplot(121)
        im_path = self.all_im_paths[self.init_frames[-1]]
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(im, cmap="grey")

        c_map = plt.get_cmap(self.c_map_name)
        sc = ax.scatter(self.keypoints[:, 0], self.keypoints[:, 1], s=4, 
                        c=self.triangulated_points[:, 2], cmap=c_map, alpha=0.5)
        cbar = fig.colorbar(sc, orientation="horizontal")
        cbar.set_label("Distance from camera, SFM units")
        
        plt.show(block=True)

        
if __name__ == "__main__":
    dl = DataLoader("kitti")
    b = Bootstrap(dl, outlier_tolerance=(15, None, 15))
    b.get_points()
    b.draw_all()
