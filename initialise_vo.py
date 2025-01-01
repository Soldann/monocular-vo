import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *
 
import matplotlib
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class DataLoader():

    def __init__(self, dataset: str, malaga_rect=True):
        """
        Retrieves the images and camera intrinsics for easy switching between
        different datasets.

        ### Parameters
        1. dataset : str
            - Either "kitti", "malaga", "parking", "own_1", or "own_2"
        2. malaga_rect : bool
            - Define whether or not the rectified images should be returned
              for the Malaga dataset. If True, load the rectified left images
              and the average intrinsics of both cameras. If False, load the
              left distorted images and the left camera's intrinsics
        """
        
        # Keep name string reference when saving files during visualisation
        self.dataset_str = dataset
        
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

            # The images have been undistorted
            self.dist = None

            # Path to the directory of the images; use image_0 by default
            self.im_dir = base_path.joinpath("05", "image_0")

            # List of paths to all the images; use image_0 by default
            self.all_im_paths = list(self.im_dir.glob("*"))

            # Number of images in the dataset
            self.n_im = len(self.all_im_paths)

            # Initial frames for bootstrapping
            self.init_frames = (0, 2)

        elif dataset == "parking":
            base_path = Path.cwd().joinpath("datasets", "parking")
            
            k_matrix_path = base_path.joinpath("K.txt")
            self.K = np.loadtxt(k_matrix_path, dtype=str)
            self.K = np.array([row[i].replace(",", "") for row in self.K for i in range(3)])
            self.K = self.K.astype(float)
            self.K = self.K.reshape(3, 3)

            self.dist = None

            self.im_dir = base_path.joinpath("images")

            self.all_im_paths = list(self.im_dir.glob("*"))

            self.n_im = len(self.all_im_paths)

            # Initial frames for bootstrapping:
            self.init_frames = (0, 5)

        elif dataset == "malaga":
            base_path = Path.cwd().joinpath("datasets", 
                                            "malaga-urban-dataset-extract-07")
            k_matrix_path = base_path.joinpath("camera_params_raw_1024x768.txt")
            k_lc = np.zeros((3, 3)).astype(str)    # left camera intrinsics
            k_rc = np.zeros((3, 3)).astype(str)
            k_lc[2, 2], k_rc[2, 2] = 1., 1.        # set bottom left element to 1
            self.dist = np.zeros(2).astype(str)    # T1, T2 distortion parameters
            with open(k_matrix_path) as txt_file:
                content = txt_file.readlines()
                
                k_lc[0, 2] = content[8][13:23]     # u0 left camera
                k_lc[1, 2] = content[9][13:23]     # v0 left camera
                k_lc[0, 0] = content[10][13:23]    # alpha_u left camera
                k_lc[1, 1] = content[11][13:23]    # alpha_v left camera

                k_rc[0, 2] = content[17][13:23]    # u0 right camera
                k_rc[1, 2] = content[18][13:23]    # v0 right camera
                k_rc[0, 0] = content[19][13:23]    # alpha_u right camera
                k_rc[1, 1] = content[20][13:23]    # alpha_v right camera

                self.dist[0] = content[12][14:28]  # T1
                self.dist[1] = content[12][29:42]  # T2
            k_lc = k_lc.astype(float)
            k_rc = k_rc.astype(float)
            k_rect = 0.5 * (k_lc + k_rc)           # the average the K'es
            self.dist = self.dist.astype(float)
            if malaga_rect:
                self.K = k_rect
                self.im_dir = base_path.joinpath("malaga-urban-dataset-"
                                                 +"extract-07_rectified_1024"
                                                 +"x768_Images")
            else:
                self.K = k_lc
                self.im_dir = base_path.joinpath("Images")

            self.all_im_paths = list(self.im_dir.glob("*_left.jpg"))
            self.n_im = len(self.all_im_paths)

            # Initial frames for bootstrapping:
            self.init_frames = (0, 3)

        elif dataset == "own_1":
            base_path = Path.cwd().joinpath("own_dataset", "ds1")
            
            # Load camera parameters
            path_calib_data = base_path.joinpath("mtx_undistorted.txt")
            K = np.loadtxt(path_calib_data, delimiter=',')
            self.K = K

            # The images have been undistorted
            self.dist = None

            # Path to the directory of the images
            self.im_dir = base_path.joinpath("images", "undistorted")

            # List of paths to all the images
            self.all_im_paths = list(self.im_dir.glob("*"))

            # Number of images in the dataset
            self.n_im = len(self.all_im_paths)

            # Initial frames for bootstrapping
            self.init_frames = (0, 10)
        
        elif dataset == "own_2":
            base_path = Path.cwd().joinpath("own_dataset", "ds2")
            
            # Load camera parameters for Kitti
            path_calib_data = base_path.joinpath("mtx_undistorted.txt")
            K = np.loadtxt(path_calib_data, delimiter=',')
            self.K = K

            # The images have been undistorted
            self.dist = None

            # Path to the directory of the images
            self.im_dir = base_path.joinpath("images", "undistorted")

            # List of paths to all the images
            self.all_im_paths = list(self.im_dir.glob("*"))

            # Number of images in the dataset
            self.n_im = len(self.all_im_paths)

            # Initial frames for bootstrapping
            self.init_frames = (0, 10)

        else:  # The input is not in ("kitti", "parking", "malaga"):
            raise ValueError("Did not specify one of 'kitti', 'parking',"
                             +" 'malaga', 'own_1', or 'own_2'.")
        

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.all_im_paths))
            return (self[i] for i in range(start, stop, step))
        else:
            img_path = str(self.all_im_paths[index])
            return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


class Bootstrap():

    def __init__(self, data_loader: DataLoader, init_frames: tuple = None,
                 outlier_tolerance=(None, None, 15)):
        """
        Get an initial point cloud

        ### Parameters
        1. data_loader : DataLoader
            - DataLoader instance initialised with one of the three datasets
        2. init_frames : Tuple[int, int], (default None)
            - Tuple of two integers that specify what images are used to
              initialise the point cloud. The integers refer to the ordering
              of the files in the directory. By default (and when None is passed)
              the suggestions by the dataloader are used (data_loader.init_frames)
        3. outlier_tolerance : tuple(x_tol: float, y_tol: float, z_tol: float)
            - For overwriting the tolerance values on utils.median_outliers()
              during the get_points() method. Pass None instead of a float 
              value to allow for any value
        """
        
        self.data_loader = data_loader
        self.all_im_paths = data_loader.all_im_paths
        self.K = data_loader.K
        self.outlier_tolerance = outlier_tolerance
        self.init_frames = data_loader.init_frames if init_frames is None else init_frames
        
        # Filled in by get_points()
        self.E = None
        self.R = None
        self.t = None 
        self.triangulated_points = None
        self.candidate_points = None
        self.keypoints = None
        self.last_img = None

        # The colour map used to represent depth
        self.c_map_name = "nipy_spectral"
        

    def get_points(self, img1_index=None, img2_index=None) -> tuple:
        """
        Get keypoints and the corresponding landmarks 

        ### Returns
        Tuple[np.array, np.array, np.array, np.aarray]
        A tuple of four numpy arrays:
            - the initial keypoints P0 (from the second initial image)
            - the corresponding landmarks X0
            - the candidate keypoints C0 (from the second image)
            - the transformation matrix T_cw, from world to camera frame
        """
        if img1_index:
            self.init_frames[0] = img1_index
        if img2_index:
            self.init_frames[1] = img2_index

        # Load images
        im1 = self.data_loader[self.init_frames[0]]
        im2 = self.data_loader[self.init_frames[1]]

        # Feature extraction
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(im1, None)
        kp2, des2 = sift.detectAndCompute(im2, None)

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
        self.E, ransac_inliers = cv2.findEssentialMat(points1, points2, self.K, method=cv2.FM_RANSAC, prob=0.999, threshold=1)
        ransac_inliers = ransac_inliers.astype(np.bool_).reshape(-1)
        self.keypoints = points2[ransac_inliers]
        self.candidate_points = points2[~ransac_inliers]

        #TODO: remove points behind the camera, too far from the mean, and too far depth wise

        _, self.R, self.t, _ = cv2.recoverPose(self.E, points1[ransac_inliers], points2[ransac_inliers], self.K)
        self.transformation_matrix = np.column_stack((self.R, self.t))

        projectionMat1 = self.K @ np.column_stack((np.identity(3), np.zeros(3)))
        projectionMat2 = self.K @ self.transformation_matrix

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

        return self.keypoints.copy(), self.triangulated_points.copy(), self.candidate_points.copy(), self.transformation_matrix.copy()

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
        im = self.data_loader[self.init_frames[-1]]
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
        ax.imshow(self.data_loader[self.init_frames[-1]], cmap="grey")

        c_map = plt.get_cmap(self.c_map_name)
        sc = ax.scatter(self.keypoints[:, 0], self.keypoints[:, 1], s=4, 
                        c=self.triangulated_points[:, 2], cmap=c_map, alpha=0.5)
        cbar = fig.colorbar(sc, orientation="horizontal")
        cbar.set_label("Distance from camera, SFM units")
        
        plt.show(block=True)

        
if __name__ == "__main__":
    pass
