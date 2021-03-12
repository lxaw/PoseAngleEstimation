from vid_img_manager import VideoImgManager
import os

class Main():

    def __init__(self):
        self.VI_M = VideoImgManager()

    def img_estimation(self,img_path):
        self.VI_M.estimate_img(img_path)

    def live_estimation(self,webcam_id=0):
        self.VI_M.estimate_vid(webcam_id)

if __name__ == "__main__":
    app = Main()
    app.img_estimation("he_did_it.png")
    #app.live_estimation(0)
