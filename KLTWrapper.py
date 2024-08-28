import numpy as np
import cv2
import itertools
import time

class KLTWrapper:
    def __init__(self):
        self.frame_count = 0
        self.win_size = 15
        self.status = 0
        self.count = 0
        self.flags = 0

        self.image = None
        self.imgPrevGray = None
        self.H = None

        self.GRID_SIZE_W = 32
        self.GRID_SIZE_H = 24
        self.MAX_COUNT = 0
        self.points0 = None
        self.points1 = None
        self.v_xy = None


    def init(self, imgGray):

        (nj, ni) = imgGray.shape

        # self.MAX_COUNT = (float(ni) / self.GRID_SIZE_W + 1.0) * (float(nj) / self.GRID_SIZE_H + 1.0)
        self.lk_params = dict(winSize=(self.win_size, self.win_size),
                         maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_MAX_ITER| cv2.TERM_CRITERIA_EPS, 20, 0.03))
        self.H = np.identity(3)
        lenI = ni / self.GRID_SIZE_W -1
        lenJ = nj / self.GRID_SIZE_H -1 
        J = np.arange(lenI*lenJ) // lenI * self.GRID_SIZE_W + self.GRID_SIZE_W / 2
        I = np.arange(lenJ*lenI) % lenJ * self.GRID_SIZE_H + self.GRID_SIZE_H / 2
        self.points0 = np.expand_dims(np.array(list(zip(J, I))), 1).astype(np.float32)
        self.frame_count +=1
        self.v_xy = np.zeros((nj, ni))



    def RunTrack(self, image, imgPrev,FG_prev, H = None):
        self.frame_count +=1
        check = False
        if H is not None:
            self.H = H
            print("use Homography")
            self.v_xy = np.zeros((image.shape[:2]))

            return None,None,0
        else:
            print("run track one")
            # if self.count > 0:
            t_st = time.time()

            self.points1, _st, _err = cv2.calcOpticalFlowPyrLK(imgPrev, image, self.points0, None, **self.lk_params)

            good1 = self.points1[_st == 1]
            good0 = self.points0[_st == 1]
            check = True
            self.v_xy = np.zeros((image.shape[:2]))
            self.count = len(good1)
            print("len good points",self.count)
            print("Time for LK ",time.time()-t_st)

        if self.count > 10:
            t_homo = time.time()
            self.makeHomoGraphy(good0, good1)
            t_trans = time.time()
            print("time find homo",t_trans-t_homo)
            temp_point = cv2.perspectiveTransform(self.points0,self.H)
            t_calv = time.time()
            print("time t_calv",t_calv-t_trans)
            #TODO recheck this with video drone_catcher_16.mp4
            v = self.points1 - temp_point
            print("compare point",v.shape)
            (nj, ni) = imgPrev.shape
            lenI = int(ni / self.GRID_SIZE_W)-1
            FG_prev = FG_prev[:-1,:-1]
            print("FG_prev",FG_prev>0)
            v_point = v.reshape(lenI,-1,2,order='F')
            print("H",self.H)
            # print("v p",v_point)
            v_x = v_point[:,:,0]
            v_y = v_point[:,:,1]

            v_xy = np.sqrt(v_x**2 + v_y**2)

            v_xy = np.pad(v_xy, [(0, 1), (0, 1)], 'constant', constant_values=0)
            # print("v xy",v_xy.shape,v_xy)
            self.v_xy = np.kron(v_xy, np.ones((self.GRID_SIZE_H, self.GRID_SIZE_W)))
            # print("v xy",v_xy.shape,v_xy)

            if self.frame_count > 2:
                print("FIlter by FG -1",self.frame_count)
                v_x= v_x[FG_prev>0]
                if v_x.shape[0]<1:
                    return good1,good0,0
                v_y= v_y[FG_prev>0]
            # print("vpoint",v_point)

            d_V = np.sqrt(v_x**2 + v_y**2)
            # d_V = d_V[d_V>0]
            print("v_x",v_x.shape,v_y.shape,d_V.shape)
            # np.set_printoptions(precision=5)
            print("dx",d_V)
            v_avg1 = np.mean(d_V)
            print("v_avg1",v_avg1)
            # print("v_avg2",d_V.mean())

            v_upper = d_V[d_V>v_avg1]
            v_avg = np.mean(v_upper)
            print("v_upper",v_avg1,v_avg)
            t_end = time.time()
            print("time run track",t_end-t_calv)
        else:
            self.H = np.identity(3)
            v_avg = 0

        print("average v",v_avg)

        if check:
            return good1, good0,v_avg
        else:
            return None,None,v_avg

    def makeHomoGraphy(self, p1, p2):
        self.H, status = cv2.findHomography(p1, p2,cv2.RANSAC, 1.0)
