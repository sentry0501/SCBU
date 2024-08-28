import numpy as np
import cv2
import KLTWrapper
import ProbModel
import time

class MCDWrapper:
    def __init__(self):
        self.imgIpl = None
        self.imgGray = None
        self.imgGrayPrev = None
        self.frm_cnt = 0
        self.lucasKanade = KLTWrapper.KLTWrapper()
        self.model = ProbModel.ProbModel()

    def init(self, image):
        self.imgGray = image
        self.imgGrayPrev = image
        self.lucasKanade.init(self.imgGray)
        self.model.init(self.imgGray)


    def run(self, frame,FG_pre=None, H_mat=None):
        self.frm_cnt += 1
        self.imgIpl = frame
        self.imgGray = frame
        self.imgGray = cv2.medianBlur(self.imgGray, 5)
        if self.imgGrayPrev is None:
            self.imgGrayPrev = self.imgGray.copy()
        t_0 = time.time()

        goodnew,goodold,avg_v = self.lucasKanade.RunTrack(self.imgGray, self.imgGrayPrev,FG_pre, H_mat)
        H_final = self.lucasKanade.H
        print("H_final: ",H_final)
        self.model.scene_condition_estimation(self.imgGray,avg_v)
        t_1 = time.time()
        self.model.motionCompensate(H_final,self.imgGray)
        t_2 = time.time()
        mask = self.model.update(frame)
        t_3 = time.time()
        self.imgGrayPrev = self.imgGray.copy()
        print("time tracking LK:", t_1-t_0)
        print("time conpensate motion:", t_2-t_1)
        print("time update bg model:", t_3-t_2)
        return mask,goodnew,goodold





