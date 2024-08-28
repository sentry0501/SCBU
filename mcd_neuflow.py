import cv2
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import torch
import numpy as np
import MCDWrapper
import time

np.set_printoptions(precision=2, suppress=True)
# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset
model = ptlflow.get_model('neuflow', pretrained_ckpt='things')
model.eval()
# torch.save(model,"neuflow_full.pth")
# exit()
# print("mode",model)
device = torch.device('cuda')
model.to(device)
cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
cv2.namedWindow("flow",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("canny",cv2.WINDOW_NORMAL)
cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
cv2.namedWindow("diff",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_o",cv2.WINDOW_NORMAL)
W,H = 320,240
vid = cv2.VideoCapture('./video/demo1.mp4')

t = np.mgrid[:H, :W]
fr_ori = np.stack((t[1], t[0]), axis=2)
print("fr_ori",fr_ori,fr_ori.shape)

_, prev_frame = vid.read()
prev_frame = cv2.resize(prev_frame,(W,H))
# A helper to manage inputs and outputs of the model
io_adapter = IOAdapter(model, prev_frame.shape[:2],cuda=True)

mcd = MCDWrapper.MCDWrapper()
isFirst = True
i = 0
count = 0
while True:
    print("open frame  -------------------------------------------",i)
    _, frame = vid.read()
    if not _:
        break

    frame = cv2.resize(frame,(W,H))
    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs([prev_frame,frame])
    i+=1
    t0 = time.time()
    # Forward the inputs through the model
    # print("input",inputs)
    predictions = model(inputs)
    print("time cal flow",time.time()-t0)
    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']

    # flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    # print("flows",flows,flows.shape)

    flow_homo = flows[0][0].permute(1, 2, 0).detach().cpu().numpy()
    flow_new = fr_ori + flow_homo
    
    flow_new = flow_new[0::10, ::10, :]
    fr_ori_ = fr_ori[0::10, ::10, :]
    print("flow_new",flow_new)
    print("fr_ori",fr_ori_)
    t1 = time.time()
    homo_mat, status = cv2.findHomography(fr_ori_.reshape(-1,2), flow_new.reshape(-1,2), cv2.RANSAC, 1.0)
    print("time find H",time.time()-t1)
    print("H",homo_mat)

    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()
    # OpenCV uses BGR format
    flow_bgr_npy = cv2.cvtColor(flow_rgb_npy, cv2.COLOR_RGB2BGR)
    flow_bgr_np = flow_bgr_npy*255
    flow_bgr_np = np.uint8(flow_bgr_np)
    # flow_gray = cv2.cvtColor(flow_bgr_np, cv2.COLOR_BGR2GRAY)
    flow_gray = cv2.GaussianBlur(flow_bgr_np, ksize=(3, 3), sigmaX=0.5) 

    edges = cv2.Canny(flow_gray,50,200)
    # Show on the screen


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(320,240))
    mask = np.zeros(gray.shape, np.uint8)
    mask_o = np.zeros(gray.shape, np.uint8)
    if (isFirst):
        mcd.init(gray)
        isFirst = False
        FG_previous = mcd.model.FG.copy()
        continue
    else:
        FG_previous = mcd.model.FG.copy()
        mask, goodnew,goodold = mcd.run(gray,FG_pre=FG_previous,H_mat=homo_mat)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dia = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # Apply erosion
    print("mask",mask)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    mask_dilate = cv2.dilate(mask_eroded, kernel, iterations=1)
    mask_eroded_o = cv2.erode(mask_o, kernel, iterations=1)
    mask_dilate_o = cv2.dilate(mask_eroded_o, kernel, iterations=1)
    # frame[mask > 0, 2] = 255
    # set the kernal
    # print(mask.max())
    print("time",time.time()-t1)
    cv2.imshow('mask', mask)
    # cv2.imshow('mask_o', mask_o)
    cv2.imshow('frame', frame)
    cv2.imshow('flow', flow_bgr_npy)
    cv2.imshow('canny', edges)
    # cv2.imshow('gray', flow_gray)
    print("frame ",count)
    cv2.waitKey(0)
    prev_frame = frame
    count +=1