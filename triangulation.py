import numpy as np
import cv2


class Triangulation:

    # Intrinsic camera matrices (example values)
    K1 = np.array([[fx1, 0, cx1],
                [0, fy1, cy1],
                [0,  0,  1]])
    K2 = np.array([[fx2, 0, cx2],
                [0, fy2, cy2],
                [0,  0,  1]])

    # Rotation and translation between cameras (extrinsics)
    R = np.array([[...], [...], [...]])  # 3x3 rotation matrix
    t = np.array([[t1], [t2], [t3]])    # 3x1 translation vector

    # Compute projection matrices for both cameras
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 at origin
    P2 = K2 @ np.hstack((R, t))                        # Camera 2

    # 2D points in both images (matched correspondences)
    points_img1 = np.array([[x1_1, y1_1], [x1_2, y1_2], ...])  # Points in image 1
    points_img2 = np.array([[x2_1, y2_1], [x2_2, y2_2], ...])  # Points in image 2

    # Convert points to homogeneous coordinates
    points_img1 = points_img1.T  # Transpose to (2, N)
    points_img2 = points_img2.T  # Transpose to (2, N)

    # Perform triangulation
    points_4D = cv2.triangulatePoints(P1, P2, points_img1, points_img2)

    # Convert homogeneous coordinates to 3D
    points_3D = points_4D[:3] / points_4D[3]

    print("Triangulated 3D points:")
    print(points_3D.T)  # Transpose back to (N, 3)
