import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# select two frames from the beginning of the dataset for initialization

image_one = "datasets/kitti/05/image_0/000000.png"
image_two = "datasets/kitti/05/image_0/000002.png"

# Load camera parameters
calib_data = np.loadtxt("datasets/kitti/05/calib.txt", dtype='str')
calib_data = calib_data[0,1:] # slice to get projection matrix for camera 0
P1 = np.array(calib_data.astype(np.float32)).reshape((3, 4))
K, R1, T1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
t1 = T1 / T1[3]

# Displaying the results
print('Intrinsic Matrix:')
print(K)
print('Rotation Matrix:')
print(R1)
print('Translation Vector:')
print(T1.round(4))


im1 = cv2.imread(image_one, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(image_two, cv2.IMREAD_GRAYSCALE)

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

img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

points1 = np.array([kp1[match[0].queryIdx].pt for match in good])
points2 = np.array([kp2[match[0].trainIdx].pt for match in good])
E, ransac_inliers = cv2.findEssentialMat(points1, points2, K, cv2.FM_RANSAC, 0.99, 2)
print(E)
_, R, t, _ = cv2.recoverPose(E, points1[ransac_inliers], points2[ransac_inliers], K)
print("R matrix")
print(R)
print("T matrix")
print(t)
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