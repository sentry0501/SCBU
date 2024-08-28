import numpy as np
import cv2
import MCDWrapper
import time
np.set_printoptions(precision=5)

ITER = "23"
result = cv2.VideoWriter(f'result/demo_{ITER}.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 30, (640,240)) 

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
cap = cv2.VideoCapture(f'./video/drone_catcher_{ITER}.mp4')
# cap = cv2.VideoCapture('/mnt/data/shared/simulation_data/drone_catcher.mp4')

mcd = MCDWrapper.MCDWrapper()
isFirst = True
print("cap",cap)
i = 0
while True:
    print("open frame  -------------------------------------------",i)
    # if i>50:
    #     break
    ret, frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]
    # new_h = int(h*640/w)
    # frame = cv2.resize(frame,(640,new_h))
    frame = cv2.resize(frame,(320,240))
    print("shape",frame.shape)
    # path = img_path.format(i)
    i+=1
    # print(path)
    # frame = cv2.imread(path)
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
        # FG_previous = mcd.model.FG.copy()
        print("FG_previous",FG_previous)
        mask, goodnew,goodold = mcd.run(gray,FG_pre=FG_previous)


    if goodnew is None:
        continue
    vector_flows = goodnew-goodold
    magnitudes = np.sqrt(np.power(vector_flows[:,1],2) + np.power(vector_flows[:,0],2))
    angles = np.arctan2(vector_flows[:,1],vector_flows[:,0]) * 180 / np.pi
    # print("magni",magnitudes)
    print("shape good",magnitudes.shape, goodold.shape)

    mag_mean = np.mean(magnitudes)
    mag_std = np.mean(magnitudes)
    mag_dis = magnitudes-mag_mean-mag_std
    # thresh_magni = (magnitudes.max()-magnitudes.min())/2

    grid_move_false = goodold[mag_dis<=0]
    # for point,magni in zip(goodold,magnitudes):
    #     if magni < thresh_magni:
    #         grid_move_false.append(point)
            
    # print("gridmove false",grid_move_false)


    v_xy_mask = mcd.lucasKanade.v_xy.copy()
    mask_v = mask.copy()
    # mask_v[mask_v>0] = 1
    #TODO recheck this
    v_mul_candidate = v_xy_mask[mask_v>0]
    print("v_xy_mask",v_xy_mask[0,0])
    print("v_mul",v_mul_candidate)
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

    contours, hierarchy = cv2.findContours(mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    print("len contour",len(contours))
    print()
    mask_are = np.zeros(gray.shape, np.uint8)
    lst_cnt = []
    if contours:
        for contour in contours:
            # (x,y,w,h) = cv2.boundingRect(contour)
            # min_x, max_x = min(x, min_x), max(x+w, max_x)
            # min_y, max_y = min(y, min_y), max(y+h, max_y)
            # if w*h > 0.1*w_f*h_f:
            #     # print("draw one")
            #     frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
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
            print("box",box)
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
    im_demo = cv2.hconcat([frame, cv2.cvtColor(mask_are, cv2.COLOR_GRAY2BGR)]) 
    cv2.imshow('demo', im_demo)
    cv2.imshow('mask_v', v_xy_mask.astype("uint8"))
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
    result.write(im_demo)
    cv2.waitKey(0)
result.release() 