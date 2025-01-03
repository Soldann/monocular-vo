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

class VoRunner():
    def __init__(self, dl, b, vo, interval=0.25):
        self.vo = vo
        self.b = b
        self.dl = dl
        self.poses = []
        self.data_queue = queue.Queue(maxsize=1)  # Only keep the latest data
        self.visualizer = None
        self.processing = False
        self.stop_event = threading.Event()
        self.interval = interval


    def process_frames(self):
        index = self.b.init_frames[1]
        for image in self.dl[index:]:

            if self.stop_event.is_set():
                break

            p_new, pi, xi, ci = self.vo.process_frame(image, debug=[])
            self.poses.append(p_new)

            try:
                self.data_queue.put((p_new, pi, xi, ci, image, index), block=False)  # Non-blocking put
            except queue.Full:
                pass  # Drop the update if the queue is full
            
            index += 1
    
    def update_visualizer(self):
        if self.visualizer is None:
            return
        
        while self.processing and not self.stop_event.is_set():
            try:
                p_new, pi, xi, ci, image, idx = self.data_queue.get(timeout=self.interval)  # Wait for new data
                self.visualizer.update_data(p_new, pi, xi, ci, image, idx)
                time.sleep(self.interval)  # Sleep briefly to avoid busy-waiting
            except queue.Empty:
                pass
    
    def run(self) -> list[np.ndarray]:
        self.processing = True
        self.poses = []
        self.process_frames()
        self.processing = False
        self.stop_event = threading.Event()
        return self.poses
    
    def run_and_visualize(self, save=False):

        # visualizer
        if self.visualizer is not None:
            self.visualizer.close_and_destroy()
            self.visualizer = None
        
        self.visualizer = DrawTrajectory(self.b, self.on_close, save=save)

        # Start the data generation in a separate thread
        data_thread = threading.Thread(target=self.run, daemon=True)
        data_thread.start()
        self.update_visualizer()
        
        print(len(self.poses))
        return self.poses

    def on_close(self, other):
        """Close event handler for the Matplotlib window."""

        print("Window is closing. Stopping all threads...")
        self.stop_event.set()  # Signal threads to stop

        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                thread.join()  # Wait for all threads to finish
        print("All threads stopped. Cleaning up...")

