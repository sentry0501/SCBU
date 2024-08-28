import numpy as np
import cv2
import MCDWrapper
import time

color = np.random.randint(0, 255, (100000, 3))
np.set_printoptions(precision=2, suppress=True)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_o",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_LFG",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_LFG_E",cv2.WINDOW_NORMAL)
cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_bfw",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_t",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_age",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_v",cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(f'./video/demo3.mp4')
# cap = cv2.VideoCapture('/mnt/data/shared/simulation_data/drone_catcher.mp4')

mcd = MCDWrapper.MCDWrapper()
isFirst = True
print("cap",cap)
i = 0
while True:
    print("open frame  -------------------------------------------",i)
    ret, frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]

    frame = cv2.resize(frame,(320,240))
    print("shape",frame.shape)
    i+=1

    h_f,w_f = frame.shape[:2]
    t1=time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if (isFirst):
        mcd.init(gray)
        isFirst = False
        FG_previous = mcd.model.FG.copy()
        continue
    else:
        print("FG_previous",FG_previous)
        imgGrayPrev = mcd.imgGrayPrev
        mask, goodnew,goodold = mcd.run(gray,FG_pre=FG_previous)

    if goodnew is None:
        continue

    v_xy_mask = mcd.lucasKanade.v_xy.copy()
    mask_v = mask.copy()

    # idx_fg = np.where(mask_v>0)
    # point_fg = np.expand_dims(np.array(list(zip(idx_fg[1], idx_fg[0]))), 1).astype(np.float32)

    # if point_fg.shape[0]<4:
    #     vmean = 0
    # else:
    #     print("point_fg",point_fg)
        
    #     point_fg_new, _st, _err = cv2.calcOpticalFlowPyrLK(imgGrayPrev, mcd.imgGray,  point_fg, None, **mcd.lucasKanade.lk_params)

    #     v_recal = point_fg_new - point_fg
    #     v_recal = np.sqrt(v_recal[:,:,0]**2 + v_recal[:,:,1]**2)
    #     print("v_recal",v_recal)
    #     vmean = v_recal.mean()
    #     var = v_recal.var()
    #     diff = (v_recal-vmean)**2/var
    #     print("v_recalmean",vmean)
    #     point_true = point_fg[diff>1.5].astype(np.uint8)
    #     print("point_true",point_true)
    #     # mask_v[mask_v>0] = 1
    #     mask[point_true[:,1],point_true[:,0]]=0

    # TODO recheck this
    v_mul_candidate = v_xy_mask[mask_v>0]
    print("v_xy_mask",v_xy_mask[0,0])
    # print("v_mul",v_mul_candidate)
    vmean = v_mul_candidate.mean()
    print("vmean",vmean)
    # v_var = np.var(v_mul_candidate)

    # L_FG = np.zeros(mask.shape)
    # L_FG[mask_v>0] = np.power((v_xy_mask[mask_v>0] - vmean),2)/v_var

    # mask[L_FG<=1]=0
    mask[v_xy_mask<=vmean]=0

    # print("v_mul",v_mul,v_xy_mask.shape)
    # mask[v_xy_mask<vmean] = 0
    # print("vmean",vmean,v_xy_mask[40,90],v_xy_mask[107,151],v_xy_mask[87,26])
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1) 
    # mask = cv2.dilate(mask, kernel, iterations=1) 

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1) 
    mask = cv2.erode(mask, kernel, iterations=1) 
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1) 
    mask = cv2.dilate(mask, kernel, iterations=1) 

    contours, hierarchy = cv2.findContours(mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    print("len contour",len(contours))
    print()
    mask_are = np.zeros(gray.shape, np.uint8)
    lst_cnt = []
    if contours:
        for contour in contours:

            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            size1 = box[0] - box[1]
            size2 = box[1] - box[2]
            # print("are",area)
            h_c = (size1[0]**2 + size1[1]**2)**0.5
            w_c = (size2[0]**2 + size2[1]**2)**0.5
            raito = h_c/w_c
            if raito<0.4 or raito > 2.5:
                continue
            # print("box",box)
            if area > 0.000*w_f*h_f:
                # print("draw one")
                lst_cnt.append(contour)
            

    mask_are = cv2.drawContours(mask_are, lst_cnt, -1, (255,255,255), -1)
    print("maske are",mask_are.shape)


    frame[mask_are > 0, 2] = 255
    FG_previous = mcd.model.rebinMax(mask,(mcd.model.GRID_SIZE_H,mcd.model.GRID_SIZE_W))
    # print(mask.max())
    print("time",time.time()-t1)
    cv2.imshow('img', frame)
    cv2.imshow('mask', mask)
    im_demo = cv2.hconcat([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]) 
    cv2.imshow('demo', im_demo)
    cv2.imshow('mask_v', v_xy_mask.astype("uint8"))
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
    # result.write(im_demo)
    cv2.waitKey(0)
# result.release() 
