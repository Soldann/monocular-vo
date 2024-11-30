import numpy as np
import cv2
import matplotlib.pyplot as plt
 
import matplotlib
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def drawCamera(ax, position, direction, length_scale = 1, head_size = 10, 
        equal_axis = True, set_ax_limits = True):
    # Draws a camera consisting of arrows into a 3d Plot
    # ax            axes object, creates as follows
    #                   fig = plt.figure()
    #                   ax = fig.add_subplot(projection='3d')
    # position      np.array(3,) containing the camera position
    # direction     np.array(3,3) where each column corresponds to the [x, y, z]
    #               axis direction
    # length_scale  length scale: the arrows are drawn with length
    #               length_scale * direction
    # head_size     controls the size of the head of the arrows
    # equal_axis    boolean, if set to True (default) the axis are set to an 
    #               equal aspect ratio
    # set_ax_limits if set to false, the plot box is not touched by the function

    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='r')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 0]],
                [position[1], position[1] + length_scale * direction[1, 0]],
                [position[2], position[2] + length_scale * direction[2, 0]],
                **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='g')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 1]],
                [position[1], position[1] + length_scale * direction[1, 1]],
                [position[2], position[2] + length_scale * direction[2, 1]],
                **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='b')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 2]],
                [position[1], position[1] + length_scale * direction[1, 2]],
                [position[2], position[2] + length_scale * direction[2, 2]],
                **arrow_prop_dict)
    ax.add_artist(a)

    if not set_ax_limits:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlim([min(xlim[0], position[0]), max(xlim[1], position[0])])
    ax.set_ylim([min(ylim[0], position[1]), max(ylim[1], position[1])])
    ax.set_zlim([min(zlim[0], position[2]), max(zlim[1], position[2])])
    
    # This sets the aspect ratio to 'equal'
    if equal_axis:
        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                       np.ptp(ax.get_ylim()),
                       np.ptp(ax.get_zlim())))



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

print(len(kp1))

img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

points1 = np.array([kp1[match[0].queryIdx].pt for match in good])
points2 = np.array([kp2[match[0].trainIdx].pt for match in good])
E, ransac_inliers = cv2.findEssentialMat(points1, points2, K, cv2.FM_RANSAC, 0.99, 2)
ransac_inliers = ransac_inliers.astype(np.bool_).reshape(-1)
print(E)
_, R, t, m = cv2.recoverPose(E, points1[ransac_inliers], points2[ransac_inliers], K)
print("R matrix")
print(R)
print("T matrix")
print(t)


projectionMat1 = np.column_stack((np.identity(3), np.zeros(3)))
projectionMat2 = np.column_stack((R,t))

print(projectionMat1)
print(projectionMat2)


triangulated_points = cv2.triangulatePoints(projectionMat1, projectionMat2, points1[ransac_inliers].T, points2[ransac_inliers].T)
triangulated_points = triangulated_points / triangulated_points[3]
triangulated_points = triangulated_points[:3, :].T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
print(triangulated_points[:,0])
ax.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], marker='o', s=5, c='r', alpha=0.5)

# Configure the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale = 2)

plt.show()

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