import numpy as np
import cv2
import MCDWrapper
import time
from ultralytics import YOLO
np.set_printoptions(precision=5)

# ITER = "23"
# result = cv2.VideoWriter(f'result/realdrone_demo.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 30, (640,240)) 
model_detect = YOLO("best.onnx")

color = np.random.randint(0, 255, (100000, 3))
np.set_printoptions(precision=2, suppress=True)
cv2.namedWindow("demo",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_o",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_LFG",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_LFG_E",cv2.WINDOW_NORMAL)
cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_bfw",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_t",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_age",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_v",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_v_0",cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture('/mnt/data/shared/simulation_data/output_2.mp4')
# cap = cv2.VideoCapture('/mnt/data/shared/simulation_data/countryside_2uav_3.avi')
# cap = cv2.VideoCapture('/mnt/data/shared/simulation_data/real_drone.MP4')
# cap = cv2.VideoCapture(f'../video/drone_catcher_16.mp4')
# cap = cv2.VideoCapture('../video/demo3.mp4')
# cap = cv2.VideoCapture('../video/woman.mp4')

lk_params = dict(winSize=(15, 15),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_MAX_ITER| cv2.TERM_CRITERIA_EPS, 10, 0.003))

mcd = MCDWrapper.MCDWrapper()
isFirst = True
print("cap",cap)
i = 0
while True:
    print("open frame  -------------------------------------------",i)
    # if i>367:
    #     break
    ret, frame = cap.read()
    if not ret:
        print("end of vid")
        break
    h,w = frame.shape[:2]
    # new_h = int(h*640/w)
    # frame = cv2.resize(frame,(640,new_h))
    t1=time.time()
    frame = cv2.resize(frame,(320,240))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("shape",frame.shape)
    # path = img_path.format(i)
    i+=1
    # if i%5 !=0:
    #     continue
    # print(path)
    # frame = cv2.imread(path)
    h_f,w_f = frame.shape[:2]
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
        imgGrayPrev = mcd.imgGrayPrev
        mask, goodnew,goodold = mcd.run(gray,FG_pre=FG_previous)

    t_post = time.time()
    v_xy_mask = mcd.lucasKanade.v_xy.copy()
    mask_v = mask.copy()

    t_lk = time.time()
    idx_fg = np.where(mask_v>0)
    point_fg = np.expand_dims(np.array(list(zip(idx_fg[1], idx_fg[0]))), 1).astype(np.float32)
    if point_fg.shape[0]<4:
        vmean = 0
    else:
        print("point_fg",point_fg.shape)
        
        point_fg_pre, _st, _err = cv2.calcOpticalFlowPyrLK( mcd.imgGray, imgGrayPrev,  point_fg, None, **lk_params)

        good_pre = point_fg_pre[_st == 1]
        good_cur = point_fg[_st == 1]
        if len(good_pre)>=4:

            good_pre_ex = np.expand_dims(good_pre,1)
            # print("good_pre_ex",good_pre_ex)
            # print("good_cur",good_cur.shape)
            print("H recal V",mcd.lucasKanade.H)
            temp_point_cur = cv2.perspectiveTransform(good_pre_ex,mcd.lucasKanade.H)
            # print("temp_point_cur",temp_point_cur)
            temp_point_cur = temp_point_cur.reshape((-1,2))
            v_point = temp_point_cur - good_cur
            v_vec = np.sqrt(v_point[:,0]**2 + v_point[:,1]**2)
            noise_good_cur = good_cur[v_vec<0.5].astype("int")
            mask[noise_good_cur[:,1],noise_good_cur[:,0]] = 0
            for tlk, po in enumerate(good_cur):
                if 187<=po[1]<=192 and 157<=po[0]<=162:
                    # print("temp_point_po",po)
                    # print("good_pre",good_pre[tlk])
                    # print("temp_point_cur",temp_point_cur[tlk])
                    print("v_vec",v_vec[tlk])
    print("time lk 2 ",time.time()-t_lk)
    cv2.imshow('mask_v_0',mask)

    # # TODO recheck this
    # v_mul_candidate = v_xy_mask[mask_v>0]
    # print("v_xy_mask",v_xy_mask[0,0])
    # # print("v_mul",v_mul_candidate)
    # vmean = v_mul_candidate.mean()
    # print("vmean",vmean)
    # # v_var = np.var(v_mul_candidate)

    # # L_FG = np.zeros(mask.shape)
    # # L_FG[mask_v>0] = np.power((v_xy_mask[mask_v>0] - vmean),2)/v_var

    # # mask[L_FG<=1]=0
    # mask[v_xy_mask<vmean]=0
    # cv2.imshow('mask_v',mask)

    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1) 
    # mask = cv2.dilate(mask, kernel, iterations=1) 


    contours, hierarchy = cv2.findContours(mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    print("len contour",len(contours))
    print()
    mask_are = np.zeros(gray.shape, np.uint8)
    lst_cnt = []
    if contours:
        for contour in contours:

            area = cv2.contourArea(contour)
            if not area or area <= 0.00*0.00*w_f*h_f or area is None:
                # print("draw one")
                continue
            # rect = cv2.minAreaRect(contour)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # size1 = box[0] - box[1]
            # size2 = box[1] - box[2]
            # # print("are",area)
            # h_c = (size1[0]**2 + size1[1]**2)**0.5
            # w_c = (size2[0]**2 + size2[1]**2)**0.5
            # raito = h_c/w_c
            # if raito<0.25 or raito > 4:
            #     continue
            # hull = cv2.convexHull(contour)

            # hull_are = np.zeros(gray.shape, np.uint8)
            # mask_hull = cv2.drawContours(hull_are, [hull], -1, (255,255,255), -1)
            # hull_area = np.where(mask_hull>0)[0].shape[0]
            # hull_are = np.zeros(gray.shape, np.uint8)
            # mask_contour = cv2.drawContours(hull_are, [contour], -1, (255,255,255), -1)
            # cnt_area = np.where(mask_contour>0)[0].shape[0]
            # print("point hull",cnt_area,hull_area)
            # if cnt_area/hull_area < 0.4:
            #     print("drop by hull")
            #     continue

            lst_cnt.append(contour)
    #         # print("box",box)
            # cv2.imshow('mask_age',hull_are)
            # if i>730:
            #     cv2.waitKey(0)
    mask = cv2.drawContours(mask_are, lst_cnt, -1, (255,255,255), -1)
    # cv2.imshow('mask_LFG',mask)
    # print("maske are",mask.shape)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1) 
    mask = cv2.erode(mask, kernel, iterations=1) 

    cv2.imshow('mask_v',mask)
    FG_previous = mcd.model.rebinMax(mask,(mcd.model.GRID_SIZE_H,mcd.model.GRID_SIZE_W))
    # print(mask.max())
    t_post_end = time.time()

    mask_drone = np.zeros(gray.shape, np.uint8)
    #TODO filter by yolodetect
    lst_drone = model_detect(frame,iou=0.5)[0]
    lst_drone = lst_drone.boxes.xyxy.cpu().numpy()
    print("lst_drone",lst_drone)
    for box in lst_drone:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask_drone[y1:y2,x1:x2] = 255
        

    #cal velocity and direction
    mask = np.bitwise_and(mask,mask_drone)
    
    contours, hierarchy = cv2.findContours(mask,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    filterd_contours = []
    #filter contour too small
    for obj in contours:
        area = cv2.contourArea(obj)
        if not area or area <= 0.005*0.005*w_f*h_f or area is None:
            print("draw one")
            continue
        filterd_contours.append(obj)
    mask_combine = np.zeros(gray.shape, np.uint8)
    mask = cv2.drawContours(mask_combine, filterd_contours, -1, (255,255,255), -1)

    lst_move_object = []
    idx_fg = np.where(mask>0)
    point_fg = np.expand_dims(np.array(list(zip(idx_fg[1], idx_fg[0]))), 1).astype(np.float32)
    if point_fg.shape[0]>=4:
        
        point_fg_pre, _st, _err = cv2.calcOpticalFlowPyrLK(mcd.imgGray,imgGrayPrev, point_fg, None, **mcd.lucasKanade.lk_params)

        good_pre = point_fg_pre[_st == 1]
        good_cur = point_fg[_st == 1]

        vector_flows = good_cur - good_pre

        # vector_flows = good_pre - good_cur
        velocity = np.sqrt(vector_flows[:,0]**2 + vector_flows[:,1]**2)
        directions = np.arctan2(-vector_flows[:,1],vector_flows[:,0]) * 180 / np.pi

        for tlk, po in enumerate(good_cur):
            if 187<=po[1]<=192 and 157<=po[0]<=162:
                print("temp_point_po",po)
                print("good_pre",good_pre[tlk])
                print("velocity",velocity[tlk])
        # print("vector_flows",vector_flows)
        # print("velocity",velocity)
        good_cur = good_cur.astype("int")
        # v_mask = np.zeros((h_f,w_f))
        v_mask = np.zeros((h_f,w_f,2))
        v_flag = np.zeros((h_f,w_f))

        dir_mask = np.zeros((h_f,w_f))
        dir_flag = np.zeros((h_f,w_f))

        # v_mask[good_cur[:,1],good_cur[:,0]] = velocity
        v_mask[good_cur[:,1],good_cur[:,0]] = vector_flows
        v_flag[good_cur[:,1],good_cur[:,0]] = 1

        dir_mask[good_cur[:,1],good_cur[:,0]] = directions
        dir_flag[good_cur[:,1],good_cur[:,0]] = 1

         
        for obj in filterd_contours:
            # print("countour",obj)
            rect =  cv2.boundingRect(obj)
            #TODO recal by all point inside contour ex. findconnectedComponent...
            obj = obj[:,0,:]
            v_mat = v_mask[obj[:,1],obj[:,0]]
            v_c = v_flag[obj[:,1],obj[:,0]]
            
            dir_mat = dir_mask[obj[:,1],obj[:,0]]
            dir_c = dir_flag[obj[:,1],obj[:,0]]

            # velo = v_mat.sum()/(v_c.sum() + 10e-6)
            print("vamt",v_mat)
            v_mat = v_mat.sum(axis=0)/(v_c.sum() + 10e-6)
            print("vamt",v_mat)
            velo =  np.sqrt(v_mat[0]**2 + v_mat[1]**2)
            print("velo",velo)
            if velo <0.5:
                continue
            dir_obj = dir_mat.sum()/(dir_c.sum()+10e-6)
            print("v_mat", v_mat)
            print("rect", rect)
            print("vel", velo)
            print("dir_obj", dir_obj)
            lst_move_object.append((rect, velo, dir_obj))

    lst_move_drone = {}
    font_scale = 1e-3*w_f

    for obj in lst_move_object:
        (rect, velo, dir_obj) = obj
        x,y,w,h = rect
        x = x+w/2
        y = y+h/2
        print("rect",x,y,w,h)
        for k,box in enumerate(lst_drone):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print("k,box",x1, y1, x2, y2)
            if x1<x<x2 and y1<y<y2:
                if str(k) not in lst_move_drone:
                    lst_move_drone[str(k)] = ((x1, y1, x2, y2), velo, dir_obj)
                else:
                    if lst_move_drone[str(k)][1]<velo:
                        lst_move_drone[str(k)] = ((x1, y1, x2, y2), velo, dir_obj)

    print("lst move drone",lst_move_drone)
    for key,val in lst_move_drone.items():
        (bbox, velo, dir_obj) = val
        x1, y1, x2, y2 = bbox
        frame = frame = cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), 2)
        frame = cv2.putText(frame, '{:.2f} {:.2f}'.format(dir_obj,velo), (x1-2,y1-2), cv2.FONT_HERSHEY_SIMPLEX , font_scale, (0,0,255), 1, cv2.LINE_AA)
    print("time post",t_post_end-t_post)
    print("time",time.time()-t1)
    cv2.imshow('img', frame)
    cv2.imshow('mask', mask)

    cv2.imshow('mask_t', mask_drone)
    im_demo = cv2.hconcat([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]) 
    cv2.imshow('demo', im_demo)

    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
    # result.write(im_demo)
    cv2.waitKey(0)
# result.release() 
