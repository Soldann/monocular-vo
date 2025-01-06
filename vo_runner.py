import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import queue
from cont_vo import VO
from initialise_vo import Bootstrap, DataLoader
from draw_trajectory import DrawTrajectory
import threading
import time
import matplotlib.pyplot as plt


class VoRunner():
    def __init__(self, dl, b, vo, interval=1):
        self.vo = vo
        self.b = b
        self.dl = dl
        self.poses = []
        self.data_queue = queue.Queue(maxsize=3000)  # Only keep the latest data

        self.visualizer = None
        self.processing = False
        self.stop_event = threading.Event()
        self.interval = interval


    def process_frames(self):
        index = self.b.init_frames[1] + 1

        for image in self.dl[self.b.init_frames[1] + 1:]:
            if self.stop_event.is_set():
                break
            
            if image is None:
                break

            optimised_poses, pi, xi, ci = self.vo.process_frame(image, debug=[])
            p_new = optimised_poses[-1]
            updated_poses = optimised_poses[:-1]
            if len(updated_poses) > 0:
                self.poses[-len(updated_poses):] = updated_poses
            self.poses.append(p_new)
            self.data_queue.put((p_new, pi, xi, ci, image, index), block=False)  # Non-blocking put

            index += 1
        
    def update_visualizer(self):
        if self.visualizer is None:
            return
        
        while self.processing and not self.stop_event.is_set():
            p_news = []
            p_new, pi, xi, ci, image, idx = None, None, None, None, None, None
            while not self.data_queue.empty():
                p_new, pi, xi, ci, image, idx = self.data_queue.get(timeout=self.interval, block=False)  # Wait for new data
                p_news.append(p_new)
            
            if len(p_news) > 0:
                self.visualizer.update_data(p_news, pi, xi, ci, image, idx)
            time.sleep(self.interval) # Sleep briefly to avoid busy-waiting

        # Final update
        p_news = []
        p_new, pi, xi, ci, image, idx = None, None, None, None, None, None
        while not self.data_queue.empty():
            p_new, pi, xi, ci, image, idx = self.data_queue.get(block=False)  # Wait for new data
            p_news.append(p_new)
        if len(p_news) > 0:
            self.visualizer.update_data(p_news, pi, xi, ci, image, idx)
    
    def run(self) -> list[np.ndarray]:
        self.stop_event = threading.Event()
        self.processing = True
        self.poses = []
        self.process_frames()
        self.processing = False
        return self.poses
    
    def run_and_visualize(self, save=False):

        # visualizer
        if self.visualizer is not None:
            self.visualizer.close_and_destroy()
            self.visualizer = None
        
        self.visualizer = DrawTrajectory(self.b, self.on_close, save=save)
        self.processing = True #ensure true due to delayed setting in run thread.
        
        # Start the data generation in a separate thread
        data_thread = threading.Thread(target=self.run, daemon=True)
        data_thread.start()
        self.update_visualizer()

        while not self.stop_event.is_set():  # Continue until the event is set
            plt.pause(0.1)


        return self.poses

    def on_close(self, other):
        """Close event handler for the Matplotlib window."""

        print("Window is closing. Stopping all threads...")
        self.stop_event.set()  # Signal threads to stop

        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                thread.join()  # Wait for all threads to finish
        print("All threads stopped. Cleaning up...")

