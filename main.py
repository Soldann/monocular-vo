import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import queue
from cont_vo import VO
from initialise_vo import Bootstrap, DataLoader
from utils import DrawTrajectory
import threading
import time

dl = DataLoader("malaga")
b = Bootstrap(dl, outlier_tolerance=(15, None, 15))
#b = Bootstrap(dl, outlier_tolerance=(500, 500, 500), init_frames=(0, 10))
vo = VO(b)
b.draw_all()

dt = DrawTrajectory(b, save=False)
data_queue = queue.Queue(maxsize=1)  # Only keep the latest data
poses = []
processing_done = False


def process_frames():
    index = b.init_frames[1]
    for image in dl[b.init_frames[1]:]:
        # debug=[VO.Debug.KLT]
        print("Frame", index)
        # if index == 28 or index == 29 or index == 30:
        #     p_new, pi, xi = vo.process_frame(image, debug=[VO.Debug.KLT])
        # else:
        p_new, pi, xi, ci = vo.process_frame(image, debug=[])
        try:
            data_queue.put((p_new, pi, xi, ci, image, index), block=False)  # Non-blocking put
        except queue.Full:
            pass  # Drop the update if the queue is full
        poses.append(p_new)
        print(p_new)
        index += 1

    processing_done = True
    
def update_plot():
    empty_queue = 0
    while not processing_done:
        try:
            p_new, pi, xi, ci, image, idx = data_queue.get(timeout=0.1)  # Wait for new data
            dt.update_data(p_new, pi, xi, ci, image, idx)
            time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
            empty_queue = 0
        except queue.Empty:
            empty_queue += 1
            if empty_queue > 10:  # If no new data for 1 second, break the loop
                    break

# Start the data generation in a separate thread
data_thread = threading.Thread(target=process_frames, daemon=True)
data_thread.start()

update_plot()

# Save the pose list 
trajectory_dir = Path.cwd().joinpath("solution_trajectories")
if not trajectory_dir.is_dir():
    trajectory_dir.mkdir()
save_path = trajectory_dir.joinpath(f"{dl.dataset_str}.pkl")
with open(save_path, "wb") as f:
    pickle.dump(poses, f)

input()
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