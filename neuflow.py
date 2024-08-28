import cv2
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import torch
import numpy as np
import time
# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset
model = ptlflow.get_model('neuflow', pretrained_ckpt='sintel')
# print("mode",model)https://www.youtube.com/watch?v=M-WOa9E--Ug
device = torch.device('cuda')
model.to(device)
cv2.namedWindow("prev_frame",cv2.WINDOW_NORMAL)
cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
cv2.namedWindow("flow",cv2.WINDOW_NORMAL)
W,H = 640,480
vid = cv2.VideoCapture('../video/dash_3.mp4')

t = np.mgrid[:H, :W]
fr_ori = np.stack((t[0], t[1]), axis=2)


_, prev_frame = vid.read()
prev_frame = cv2.resize(prev_frame,(W,H))
# A helper to manage inputs and outputs of the model
io_adapter = IOAdapter(model, prev_frame.shape[:2],cuda=True)
while True:
    _, frame = vid.read()
    if not _:
        break

    frame = cv2.resize(frame,(W,H))
    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs([prev_frame,frame])

    t0 = time.time()
    # Forward the inputs through the model
    predictions = model(inputs)
    print("time",time.time()-t0)
    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']

    # flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    print(flows)
    flow_homo = flows[0][0].permute(1, 2, 0).detach().cpu().numpy()
    flow_new = fr_ori + flow_homo
    
    flow_new = flow_new[0::16, ::16, :]
    fr_ori_ = fr_ori[0::16, ::16, :]
    print("flow_new",flow_new.shape)
    print("fr_ori",fr_ori_.shape)
    t1 = time.time()
    homo_mat, status = cv2.findHomography(flow_new.reshape(-1,2), fr_ori_.reshape(-1,2), cv2.RANSAC, 1.0)
    print("time find",time.time()-t1)
    print("H",homo_mat)

    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()
    # OpenCV uses BGR format
    flow_bgr_npy = cv2.cvtColor(flow_rgb_npy, cv2.COLOR_RGB2BGR)

    # Show on the screen
    cv2.imshow('prev_frame', prev_frame)
    cv2.imshow('frame', frame)
    cv2.imshow('flow', flow_bgr_npy)
    cv2.waitKey(10)
    prev_frame = frame