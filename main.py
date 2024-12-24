import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *
 
from cont_vo import VO
from initialise_vo import Bootstrap, DataLoader


dl = DataLoader("parking")
b = Bootstrap(dl, outlier_tolerance=(15, None, 15))
vo = VO(b)
b.draw_all()
print(b.transformation_matrix)

path = []
fig = plt.figure(figsize=(14, 5))
ax_3d = fig.add_subplot(122, projection='3d')
ax_img = fig.add_subplot(121)

for image in dl[dl.init_frames[1]:]:
    new_transform = vo.process_frame(image, debug=[VO.Debug.TRIANGULATION])

    path.append(inverse_transformation(new_transform))

    np_path = np.array(path)

    ax_img.clear()
    ax_3d.clear()

    # plot path
    ax_3d.view_init(elev=-90, azim=0, roll=-90)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.scatter(vo.Xi_1[:, 0], vo.Xi_1[:, 1], 
            vo.Xi_1[:, 2], marker='o', s=5, 
            c="red", alpha=0.5)
    ax_3d.plot(np_path[:,0,3], np_path[:,1,3], 
            np_path[:,2,3], marker='o', 
            c="blue", alpha=0.5)
    # plot image
    ax_img.imshow(image, cmap="grey")

    c_map = plt.get_cmap("nipy_spectral")
    sc = ax_img.scatter(vo.Pi_1[:, 0], vo.Pi_1[:, 1], s=4, 
                    c="red", alpha=0.5)
    sc = ax_img.scatter(vo.Ci_1[:, 0], vo.Ci_1[:, 1], s=4, 
                    c="green", alpha=0.5)


    print(new_transform)
    plt.pause(1)
# vo.next_image()
# vo.track_keypoints()
# vo.draw_keypoint_tracking()

"""
dl = DataLoader("kitti")
b = Bootstrap(dl)
bootstrap_keypoints, bootstrap_3d_points, bootstrap_candidate_points = b.get_points()

prevImg = cv2.imread(dl.all_im_paths[2].__str__(), cv2.IMREAD_GRAYSCALE)
nextImg = cv2.imread(dl.all_im_paths[3].__str__(), cv2.IMREAD_GRAYSCALE)

# # Example points (replace with your actual points)
# prevPts = np.array([[100, 200], [150, 250]], dtype=np.float32)
# nextPts = np.array([[110, 210], [160, 260]], dtype=np.float32)

# # Ensure points are not empty
# if prevPts.size == 0 or nextPts.size == 0:
#     raise ValueError("Input points are empty")

# # Convert points to the correct data type
# prevPts = prevPts.astype(np.float32)
# nextPts = nextPts.astype(np.float32)

# # Use the points in the calc function
# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 0,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# nextpt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, None, **lk_params)

vo = VO(b.K, bootstrap_keypoints, bootstrap_3d_points, bootstrap_candidate_points, None, None)
p_new = vo.process_frame(prevImg, nextImg, debug=True)

# Drawing the new keypoints:
fig, ax = plt.subplots()
ax.imshow(nextImg, cmap="grey")
sc = ax.scatter(p_new[:, 0], p_new[:, 1], s=4, alpha=0.5)
plt.show(block=True)

# # Plot the results
# plt.figure()
# dh = int(im2.shape[0] - im1.shape[0])
# top_padding = int(dh/2)
# img1_padded = cv2.copyMakeBorder(im1, top_padding, dh - int(dh/2),
#         0, 0, cv2.BORDER_CONSTANT, 0)
# plt.imshow(np.c_[img1_padded, im2], cmap = "gray")

# print(kp1)

# for match in good:
#     img1_idx = match.queryIdx
#     img2_idx = match.trainIdx
#     x1 = kp1[img1_idx].pt[1]
#     y1 = kp1[img1_idx].pt[0] + top_padding
#     x2 = kp2[img2_idx].pt[1] + im1.shape[1]
#     y2 = kp2[img2_idx].pt[0]
#     plt.plot(np.array([x1, x2]), np.array([y1, y2]), "o-")
# plt.show()

# plt.imshow(im1)
# plt.show()
# plt.imshow(im2)
# plt.show()
"""