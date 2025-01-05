import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from pathlib import Path
import cv2
from arrow_3d import Arrow3D
from utils import drawCamera

class DrawTrajectory():
    def __init__(self, b, on_close, save=False):
        """
        Initialise with the bootstrap

        ### Parameters
        1. b : initialise_vo.Bootstrap
            - Bootstrapping instance that provides:
                - t : np.array. The estimated pose of the camera at the last 
                bootstrapping frame
                - p : np.array. The keypoints in P after bootstrapping
                - x : np.array. The landmarks in X after bootstrapping
                - im : np.array. Provided through b's dataloader; the last
                bootstrapping frame
        2. save : bool
            - Specify whether the plots are to be saved in a dedicated folder
            for later in-depth inspection. When save is true, the plots are
            not displayed.
        """

        # Get information from Bootstrapping object
        p = b.keypoints
        c = []
        x = b.triangulated_points
        self.frame = b.init_frames[-1]
        im = b.data_loader[self.frame]

        c_map = plt.get_cmap("plasma")  # cmap style (if mark_depth)

        # Computing the location of the camera, saving the x and z components
        c_R_cw = b.transformation_matrix[:, :3]
        c_t_cw = b.transformation_matrix[:, -1]
        w_t_wc = -1 * c_R_cw.T @ c_t_cw
        self.w_t_wc_x = [w_t_wc[0]]
        self.w_t_wc_y = [w_t_wc[1]]
        self.w_t_wc_z = [w_t_wc[2]]

        # Computing the depth component
        z_c = (x @ c_R_cw.T)[:, 2]  # compute z_c

        # Setting the directories for saving if desired
        self.save = save
        if self.save:
            plot_dir_general = Path.cwd().joinpath("vo_plots")
            if not plot_dir_general.is_dir():
                plot_dir_general.mkdir()
            dataset_str = b.data_loader.dataset_str
            self.plot_dir_dataset = plot_dir_general.joinpath(dataset_str)
            if not self.plot_dir_dataset.is_dir():
                self.plot_dir_dataset.mkdir()

        # Plot settings
        plt.ion()
        self.fig, axs = plt.subplot_mosaic(
            [["u", "u"], [ "l", "r"]],
            figsize=(10, 8),                     # (width x height)
            layout="constrained"
        )
        self.fig.canvas.mpl_connect('close_event', on_close)
        self.u = axs["u"]
        self.l = self.fig.add_subplot(2, 2, 3, projection="3d")

        #self.l = axs["l"]
        self.r = axs["r"]
        #self.m = axs["m"]
        self.fig.suptitle(f"Image i = {self.frame}")

        # Upper plot:
        self.u_im = self.u.imshow(im, cmap="grey")
        self.u_candidatepoints = self.u.scatter(p[:, 0], p[:, 1], marker="x",
                                          alpha=0.5, c='g', s=25)
        self.u_keypoints = self.u.scatter(p[:, 0], p[:, 1], marker="o",
                                          alpha=1, cmap=c_map, c=z_c, s=5)
        self.cbar = self.fig.colorbar(self.u_keypoints, orientation="vertical")
        self.cbar.set_label("Distance from camera, SFM units")
        self.u.set_title("Current image with landmarks")

        # The area where candidates are sampled:
        w, h = im.shape
        c = [h/2, w/2]
        rect_w, rect_h = 0.2 * w, 0.2 * h
        corners = np.array([[c[0] + 10, c[1] + 10],
                            [c[0] + 10, c[1] - 10],
                            [c[0] - 10, c[1] - 10],
                            [c[0] - 10, c[1] + 10]])
        self.rect = Polygon(corners, closed=True, edgecolor="g", facecolor="none",
                            linewidth=3)
        self.u.add_patch(self.rect)

        # Right: xz landmark graphic
        self.r_landmarks = self.r.scatter(x[:, 0], x[:, 2], marker="o",
                                          alpha=0.5, cmap=c_map, c=z_c)
        self.r_prev_pos = self.r.plot(0, 0, marker="x", c="b", markersize=5,
                                      linestyle=None)[0]
        self.r_pos = self.r.plot(w_t_wc[0], w_t_wc[2], c="b", marker="X",
                                 markersize=10)[0]
        self.r.set_xlabel("$X_w$ - SFM units")
        self.r.set_ylabel("$Z_w$ - SFM units")
        self.r.set_ylim(-200, 200)
        self.r.autoscale(False)
        self.r.grid(visible=True, axis="both")
        self.r.set_title("Landmarks $P$ in world frame")

        # Left: xz camera position
        self.l_pos = self.l.plot(self.w_t_wc_x, self.w_t_wc_y, self.w_t_wc_z)[0]
        drawCamera(self.l, w_t_wc, c_R_cw)
        self.l.set_xlabel("$X_w$ - SFM units")
        self.l.set_ylabel("$Z_w$ - SFM units")
        self.l.set_zlabel("$Y_w$ - SFM units")
        self.l.set_xlim(5, 5)
        self.l.set_ylim(5, 5)
        self.l.set_zlim(5, 5)
        self.l.autoscale(False)


        #self.l.grid(visible=True)
        # self.l.set_xlabel("Camera poses in world frame")
        # self.l.set_xlabel("$X_w$ - SFM units")
        # self.l.set_ylabel("$Z_w$ - SFM units")


        if self.save:
            self.fig.savefig(self.plot_dir_dataset.joinpath(f"{self.frame:0{4}}.jpg"))
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def update_data(self, t: np.array, p: np.array, x: np.array, c: np.array,
                    im: np.array, idx: int, extremes: np.array):
        """
        Update the plot

        ### Parameters
        1. t : np.array
            - Latest transformation matrix from homogeneous world coordinates
            to (X, Y, Z) in camera coordinates. A (3 x 4) matrix
        2. p : np.array
            - The current keypoints in set Pi belonging to the landmarks
            xi. Shape (n x 2)
        3. x : np.arrays
            - Np.array containing the landmarks currently used for camera
            localisation on its rows. (n x 3). The landmarks are given in
            the world coordinate system.
        4. im : np.array
            - Image
        """


        ### ------- PROCESS DATA ------- ###
        
        # Computing the current location of the camera, saving x and z comp.
        c_R_cw = t[:, :3]
        c_t_cw = t[:, -1]
        w_t_wc = -1 * c_R_cw.T @ c_t_cw
        self.w_t_wc_x.append(w_t_wc[0])
        self.w_t_wc_z.append(w_t_wc[2])
        self.w_t_wc_y.append(w_t_wc[1])

        corners = np.array([[extremes[0, 0], extremes[1, 1]],
                            [extremes[0, 1], extremes[1, 1]],
                            [extremes[0, 1], extremes[1, 0]],
                            [extremes[0, 0], extremes[1, 0]]])
        self.rect.set_xy(corners)

        # Computing the depth component
        z_c = (x @ c_R_cw.T)[:, 2]  # compute z_c

        ### ------- UPDATE FIGURE DATA ------- ###

        # Upper plot
        self.u_im.set(data=im)
        self.u_keypoints.set_offsets(p)
        self.u_candidatepoints.set_offsets(c)
        self.r_landmarks.set_array(z_c)  # Upadte the colour mapping
        self.u_keypoints.set_array(z_c)
        self.rect.set_xy(corners)

        # Right plot
        self.r_pos.set_data([w_t_wc[0]], [w_t_wc[2]])
        self.r_landmarks.set_offsets(np.c_[x[:, 0], x[:, 2]])
        self.r_landmarks.set_array(z_c)  # Upadte the colour mapping
        self.r_prev_pos.set_data(self.w_t_wc_x, self.w_t_wc_z)

        # Left plot
        self.l_pos.set_data(self.w_t_wc_x, self.w_t_wc_y)
        self.l_pos.set_3d_properties(self.w_t_wc_z)
        for artist in self.l.get_children():
            if isinstance(artist, Arrow3D):  # Check if it's a Line2D object
                artist.remove()


        ### ------- UPDATE FIGURE SCALE ------- ###

        # Left plot
        # Compute the limits for the plot; same scale for both axes
        l_xmin, l_xmax = min(self.w_t_wc_x) - 1, max(self.w_t_wc_x) + 1
        l_zmin, l_zmax = min(self.w_t_wc_z) - 1, max(self.w_t_wc_z) + 1
        l_ymin, l_ymax = min(self.w_t_wc_y) - 1, max(self.w_t_wc_y) + 1

        max_range = max(l_xmax - l_xmin, l_zmax - l_zmin, l_ymax - l_ymin)
        mid_x = 0.5 * (l_xmin + l_xmax)
        mid_z = 0.5 * (l_zmin + l_zmax)
        mid_y = 0.5 * (l_ymin + l_ymax)

        self.l.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
        self.l.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
        self.l.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)
        drawCamera(self.l, w_t_wc, c_R_cw.T, length_scale=max_range/5)

        # Right plot
        # Making sure the landmark positions are visible:
        r_xmax = max(x[:, 0].max(), w_t_wc[0]) + 10
        r_xmin = min(x[:, 0].min(), w_t_wc[0]) - 10
        r_zmax = max(x[:, 2].max(), w_t_wc[2]) + 10
        r_zmin = min(x[:, 2].min(), w_t_wc[2]) - 10
        self.r.set_xlim(r_xmin, r_xmax)
        self.r.set_ylim(r_zmin, r_zmax)

        # Update the colour bar scale
        norm = Normalize(vmin=0, vmax=z_c.max())
        self.u_keypoints.set_norm(norm)
        self.r_landmarks.set_norm(norm)
        self.cbar.update_normal(self.u_keypoints)

        ### ------- UPDATE PLOTS ------- ###

        self.fig.suptitle(f"Image i = {idx}")
        if self.save:
            self.fig.savefig(self.plot_dir_dataset.joinpath(f"{self.frame:0{4}}.jpg"))
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def close_and_destroy(self):
        plt.close('all')
        del self.fig
        del self.ax
        del self.r
        del self.u_keypoints
        del self.r_landmarks
        del self.cbar
        gc.collect()
