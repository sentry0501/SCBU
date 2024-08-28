import numpy as np
import cv2
import itertools
import time

class ProbModel:
    def __init__(self):

        self.BLOCK_SIZE	= 4
        self.MAX_BG_AGE = 30 #15
        self.VAR_DEC_RATIO = 0.1
        self.VAR_MIN_NOISE_T = 50.0*50.0
        self.T_high = 3.5
        self.T_low = 2
        self.INIT_BG_VAR = 20.0*20.0
        self.MIN_BG_VAR = 5 * 5
        self.means = None
        self.vars = None
        self.ages = None
        self.c_count = None
        self.temp_means = None
        self.temp_vars = None
        self.temp_ages = None
        self.modelWidth = None
        self.modelHeight = None
        self.obsWidth = None
        self.obsHeight = None
        self.FG = None
        self.average_v = None
        self.b_illu = None
        self.GRID_SIZE_W = 32
        self.GRID_SIZE_H = 24

    def init(self, gray):
        gray = np.array(gray)

        (self.obsHeight, self.obsWidth) = gray.shape
        (self.modelHeight, self.modelWidth) = (self.obsHeight//self.BLOCK_SIZE, self.obsWidth//self.BLOCK_SIZE)
        # self.means = np.zeros((self.modelHeight, self.modelWidth))
        # self.vars = np.zeros((self.modelHeight, self.modelWidth))
        self.means = self.rebin(gray, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        #TODO init vars
        bigMean = np.kron(self.means, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        diff = np.power(gray - bigMean,2)
        self.vars = self.rebin(diff, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        self.ages = np.zeros((self.modelHeight, self.modelWidth))
        self.c_count = 1
        self.FG = np.zeros((self.modelHeight, self.modelWidth))
        self.average_v = 0
        self.b_illu = 0

        self.I = np.array([range(self.modelWidth)]*self.modelHeight).flatten()
        self.J = np.repeat(np.array(range(self.modelHeight)), self.modelWidth)
        self.center_points = np.asarray([self.I*self.BLOCK_SIZE+self.BLOCK_SIZE/2, self.J*self.BLOCK_SIZE + self.BLOCK_SIZE/2, np.ones(len(self.I))])

        # self.temp_means = np.zeros((self.modelHeight, self.modelWidth))
        # self.temp_vars = np.zeros((self.modelHeight, self.modelWidth))
        self.temp_means = self.rebin(gray, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        #TODO init vars
        bigMean = np.kron(self.means, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        diff = np.power(gray - bigMean,2)
        self.temp_vars = self.rebin(diff, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        self.temp_ages = np.zeros((self.modelHeight, self.modelWidth))
        H = np.identity(3)
        # self.scene_condition_estimation(gray,avg_v=10)
        # self.motionCompensate(H,gray)
        
        # self.update(gray)


    def rebin(self, arr, factor):

        f = (np.asarray(factor) - arr.shape) % factor

        temp = np.pad(arr, ((0, f[0]), (0, f[1])), 'edge')
        sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
        res = temp.reshape(sh).mean(-1).mean(1)
        return res[:res.shape[0] - f[0], : res.shape[1] - f[1]]

    def rebinMax(self, arr, factor):
        f = (np.asarray(factor) - arr.shape) % np.asarray(factor)
        temp = np.pad(arr, ((0, f[0]), (0, f[1])), 'edge')
        sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
        res = temp.reshape(sh).max(-1).max(1)
        return res[:res.shape[0] - f[0], : res.shape[1] - f[1]]

    def scene_condition_estimation(self,gray, avg_v):

       
        self.b_illu = np.mean(gray) - np.mean(self.means)
        self.average_v = avg_v
        return

    def motionCompensate(self, H,gray):
        # warping background model
        

        #invert warping centerpoint t to t-1
        H_inver = np.linalg.inv(H)
        print("HINVERT: ", H_inver)
        print("center_points: ", self.center_points)
        warp_centers = H_inver.dot(self.center_points)
        print("warp_centers",warp_centers)
        NewW = warp_centers[2, :]
        NewX = (warp_centers[0, :]/NewW)
        NewY = (warp_centers[1, :]/NewW)

        #new index of grid from t to t-1
        NewI = NewX / self.BLOCK_SIZE
        NewJ = NewY / self.BLOCK_SIZE

        idxNewI = np.floor(NewI).astype(int)
        idxNewJ = np.floor(NewJ).astype(int)

        Di = NewI - idxNewI - 0.5
        Dj = NewJ - idxNewJ - 0.5

        aDi = abs(Di)
        aDj = abs(Dj)

        M = self.means
        V = self.vars
        A = self.ages

        W_H = (aDi * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)
        W_V = (aDj * (1 - aDi)).reshape(self.modelHeight, self.modelWidth)
        W_HV = (aDi * aDj).reshape(self.modelHeight, self.modelWidth)
        W_self = ((1-aDi) * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)

        W = np.zeros((self.modelHeight, self.modelWidth))

        curMean = self.rebin(gray, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        self.temp_means = curMean.copy()
        tempMean = np.zeros(self.means.shape)
        tempAges = np.zeros(self.means.shape)

        NewI_H = idxNewI + np.sign(Di).astype(int)
        condH = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)

        t_motin = time.time()
        tempMean[self.J[condH], self.I[condH]] = W_H[self.J[condH], self.I[condH]] * M[idxNewJ[condH], NewI_H[condH]]
        print("tempMeanH", tempMean[2,45])
        tempAges[self.J[condH], self.I[condH]] = W_H[self.J[condH], self.I[condH]] * A[idxNewJ[condH], NewI_H[condH]]
        W[self.J[condH], self.I[condH]] += W_H[self.J[condH], self.I[condH]]

        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condV = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        tempMean[self.J[condV], self.I[condV]] += W_V[self.J[condV], self.I[condV]] * M[NewJ_V[condV], idxNewI[condV]]
        print("tempMeanV", tempMean[2,45])
        tempAges[self.J[condV], self.I[condV]] += W_V[self.J[condV], self.I[condV]] * A[NewJ_V[condV], idxNewI[condV]]
        W[self.J[condV], self.I[condV]] += W_V[self.J[condV], self.I[condV]]

        NewI_H = idxNewI + np.sign(Di).astype(int)
        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condHV = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        tempMean[self.J[condHV], self.I[condHV]] += W_HV[self.J[condHV], self.I[condHV]] * M[NewJ_V[condHV], NewI_H[condHV]]
        print("tempMeanHV", tempMean[2,45])
        tempAges[self.J[condHV], self.I[condHV]] += W_HV[self.J[condHV], self.I[condHV]] * A[NewJ_V[condHV], NewI_H[condHV]]
        W[self.J[condHV], self.I[condHV]] += W_HV[self.J[condHV], self.I[condHV]]

        condSelf = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        tempMean[self.J[condSelf], self.I[condSelf]] += W_self[self.J[condSelf], self.I[condSelf]] * M[idxNewJ[condSelf], idxNewI[condSelf]]
        print("tempMeanS", tempMean[2,45])
        tempAges[self.J[condSelf], self.I[condSelf]] += W_self[self.J[condSelf], self.I[condSelf]] * A[idxNewJ[condSelf], idxNewI[condSelf]]
        W[self.J[condSelf], self.I[condSelf]] += W_self[self.J[condSelf], self.I[condSelf]]
        print("time compensate",time.time()-t_motin)
        print("pre temp mean",self.temp_means[2,45], W[2,45],tempMean[2,45])
        print("pre index",idxNewI.reshape((60,80))[2,45],idxNewJ.reshape((60,80))[2,45],NewI_H.reshape((60,80))[2,45],NewJ_V.reshape((60,80))[2,45])


        # print("pre mean", M[idxNewJ[condH][630], NewI_H[condH][630]])
        # print("pre mean", M[NewJ_V[condV][630], idxNewI[condV][630]])
        # print("pre mean", M[NewJ_V[condHV][630], NewI_H[condHV][630]])
        # print("pre mean", M[idxNewJ[condSelf][630], idxNewI[condSelf][630]])
        # print("pre var", V[idxNewJ[condH][630], NewI_H[condH][630]])
        # print("pre var", V[NewJ_V[condV][630], idxNewI[condV][630]])
        # print("pre var", V[NewJ_V[condHV][630], NewI_H[condHV][630]])
        # print("pre var", V[idxNewJ[condSelf][630], idxNewI[condSelf][630]])
        # print("pre W", W_H[self.J[condH][630], self.I[condH][630]])
        # print("pre W", W_V[self.J[condV][630], self.I[condV][630]])
        # print("pre W", W_HV[self.J[condHV][630], self.I[condHV][630]])
        # print("pre W", W_self[self.J[condSelf][630], self.I[condSelf][630]])

        print("new XY", NewX.reshape((60,80))[2,45], NewY.reshape((60,80))[2,45])
        print("gray",gray[8:12,184:188])
        print("gray",gray[12:16,180:184])
        print("gray",gray[12:16,184:188])
        print("gray",gray[8:12,180:184])
        print("pre mean", M[2, 46])
        print("pre mean", M[3, 45])
        print("pre mean", M[3, 46])
        print("pre mean", M[2, 45])
        print("pre var", V[2, 46])
        print("pre var", V[3, 45])
        print("pre var", V[3, 46])
        print("pre var", V[2, 45])
        print("pre W", W_H[2,46])
        print("pre W", W_V[3,45])
        print("pre W", W_HV[3,46])
        print("pre W", W_self[2,45])
        self.temp_means[W != 0] = 0
        self.temp_ages[:] = 0
        W[W == 0] = 1
        print("Mean <0",M[M<0],self.b_illu,np.where(M<0))
        tempMean = tempMean
        self.temp_means += tempMean / W  + self.b_illu
        print("selftempmean<00000",self.temp_means[2,45],tempMean[2,45],W[2,45])
        self.temp_ages += tempAges / W

        temp_var = np.zeros(self.means.shape)

        temp_var[self.J[condH], self.I[condH]] += W_H[self.J[condH], self.I[condH]] * V[idxNewJ[condH], NewI_H[condH]]
        print("temp varH: ", temp_var[2,45])

        temp_var[self.J[condV], self.I[condV]] += W_V[self.J[condV], self.I[condV]] * V[NewJ_V[condV], idxNewI[condV]]
        print("temp varV: ", temp_var[2,45])

        temp_var[self.J[condHV], self.I[condHV]] += W_HV[self.J[condHV], self.I[condHV]] * V[NewJ_V[condHV], NewI_H[condHV]]
        print("temp varHV: ", temp_var[2,45])

        temp_var[self.J[condSelf], self.I[condSelf]] += W_self[self.J[condSelf], self.I[condSelf]] * V[idxNewJ[condSelf], idxNewI[condSelf]]
        print("temp var self: ", temp_var[2,45])
        print("self temp var: ", self.temp_vars[2,45])

        if self.c_count*self.average_v >= self.BLOCK_SIZE:
        # if True:
            print("UPDATE VAR BY COUNT")
            temp_var[self.J[condH], self.I[condH]] += W_H[self.J[condH], self.I[condH]] * np.power(self.temp_means[self.J[condH], self.I[condH]] - self.means[idxNewJ[condH], NewI_H[condH]],2)

            temp_var[self.J[condV], self.I[condV]] += W_V[self.J[condV], self.I[condV]] * np.power(self.temp_means[self.J[condV], self.I[condV]] - self.means[NewJ_V[condV], idxNewI[condV]], 2)
            
            temp_var[self.J[condHV], self.I[condHV]] += W_HV[self.J[condHV], self.I[condHV]] * np.power(self.temp_means[self.J[condHV], self.I[condHV]] - self.means[NewJ_V[condHV], NewI_H[condHV]], 2)

            temp_var[self.J[condSelf], self.I[condSelf]] += W_self[self.J[condSelf], self.I[condSelf]] * np.power(self.temp_means[self.J[condSelf], self.I[condSelf]] - self.means[idxNewJ[condSelf], idxNewI[condSelf]], 2)

        self.temp_vars[W != 0] = 0
        self.temp_vars += temp_var / W
        cond = (idxNewJ < 1) | (idxNewJ >= self.modelHeight - 1) | (idxNewI < 1) | (idxNewI >= self.modelWidth - 1)

        self.temp_vars[self.J[cond], self.I[cond]] = self.INIT_BG_VAR
        self.temp_vars[self.temp_vars < self.MIN_BG_VAR] = self.MIN_BG_VAR
        print("self temp var: ", self.temp_vars[2,45])


        # #TODO update ages 
        # check_var = self.temp_vars[:, :] - self.VAR_MIN_NOISE_T
        # check_var = np.where(check_var<0,0.0,check_var)
        # self.temp_ages[:, :] = self.temp_ages[:, :]*np.exp(-self.VAR_DEC_RATIO*check_var)
        self.temp_ages[self.J[cond], self.I[cond]] = 0

    def update(self, gray):

        print("UPDATED")
        curMean = self.rebin(gray, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        print()
        if not self.c_count*self.average_v >= self.BLOCK_SIZE:
            print("NOT UPDATE BY COUNTTTTTTTTTT")


            self.means = self.temp_means
            bigMean = np.kron(self.means, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
            (a, b) = (gray.shape[0] - bigMean.shape[0], gray.shape[1] - bigMean.shape[1])
            bigMean = np.pad(bigMean, ((0, a), (0, b)), 'edge')
            self.vars = self.temp_vars.copy()

            #TODO set alpha = 1 vs edge

            alpha = self.temp_ages / (self.temp_ages + 1)
            alpha[self.temp_ages < 1] = 0

            bigMean_ = np.kron(curMean, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
            diff = np.power(gray - bigMean_,2)
            self.vars[self.temp_ages < 1] = self.rebin(diff, (self.BLOCK_SIZE, self.BLOCK_SIZE))[self.temp_ages < 1]

            # maxes = self.rebinMax(np.power(gray - bigMean, 2), (self.BLOCK_SIZE, self.BLOCK_SIZE))
            # self.vars[self.temp_ages < 1] = self.temp_vars[self.temp_ages < 1] * alpha[self.temp_ages < 1] + (1 - alpha[self.temp_ages < 1]) * maxes[self.temp_ages < 1]
            
            self.c_count +=1 
        else:
        # if True:
            print("UPDATE BY CCOUNTTTTTTTT")
            alpha = self.temp_ages / (self.temp_ages + 1)
            alpha[self.temp_ages < 1] = 0
            self.means = (self.temp_means)* alpha + curMean * (1 - alpha)
            bigMean = np.kron(self.means, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
            (a, b) = (gray.shape[0] - bigMean.shape[0], gray.shape[1] - bigMean.shape[1])
            bigMean = np.pad(bigMean, ((0, a), (0, b)), 'edge')
            maxes = self.rebinMax(np.power(gray - bigMean, 2), (self.BLOCK_SIZE, self.BLOCK_SIZE))
            self.vars = self.temp_vars * alpha + (1 - alpha) * maxes
            self.c_count = 1
            print("self var",self.vars[2,45],self.temp_vars[2,45],maxes[2,45],alpha[2,45])

        self.vars[(self.vars < self.INIT_BG_VAR) & (self.ages == 0)] = self.INIT_BG_VAR
        self.vars[(self.vars < self.MIN_BG_VAR)] = self.MIN_BG_VAR

        self.ages = self.temp_ages.copy()
        self.ages += 1
        self.ages[(self.ages > self.MAX_BG_AGE)] = self.MAX_BG_AGE

        print("cur mean",curMean[2,45])
        print("self mean",self.means[2,45])
        print("self age",self.ages[2,45])
        print("self var",self.vars[2,45])
        print("self mean <0",self.means[self.means<0],self.b_illu,np.where(self.means<0))
        bigAges = np.kron(self.ages, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        bigVars = np.kron(self.vars, np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        
        bigAges = np.pad(bigAges, ((0, a), (0, b)), 'edge')
        bigVars = np.pad(bigVars, ((0, a), (0, b)), 'edge')
        print("bigmean",bigMean[8:12,180:184])
        print("gray",gray[8:12,180:184])
        print("big var",bigVars[8:12,180:184])
        self.distImg = np.power(gray - bigMean, 2)
        L_FG = self.distImg / bigVars
        print("L_FG",L_FG[8:12,180:184])
     
        #TODO watershed here
        out_w = np.zeros(gray.shape).astype(np.uint8)
        out_w[(bigAges > 1) & (L_FG>=self.T_high)] = 255
        out_w[(bigAges > 1) & (L_FG<=self.T_low)] = 127
        out_w[(bigAges <= 3)] = 127

        cv2.imshow("mask_bfw",out_w.astype("uint8"))
        cv2.imshow("mask_age",bigAges.astype("uint8"))
        print("out_w",out_w[8:12,180:184])
        # L_FG_img = L_FG.astype("uint8")
        # scale = 255/L_FG_img.max()
        # print("scale",scale)
        # L_FG_img = (L_FG_img*scale).astype("uint8")
        # print("max",L_FG_img.max())
        # cv2.imshow("mask_LFG",L_FG.astype("uint8"))
        # # cv2.imshow("mask_LFG_E",L_FG_img)
        # # print("mask_LFG_E",L_FG_img>127)
        # L_FG_img[L_FG_img<127] = 0
        # L_FG_img[L_FG_img>=127] = 255
        # cv2.imshow("mask_t",L_FG_img)
        img_ws = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # out_w = np.expand_dims(out_w,axis=2)
        out_w = out_w.astype("int32")
        out_w = cv2.watershed(img_ws,out_w)
        out_w = np.array(out_w)
        out_w[out_w==127]=0
        out_w[out_w == -1] = 0

        print("out_w",out_w[8:12,180:184])

        cv2.imshow("mask_o",out_w.astype("uint8"))
        # cv2.imshow("gray",gray)
        # cv2.waitKey(0)

        return out_w.astype("uint8")
