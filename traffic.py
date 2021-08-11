
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform,VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
def detect(frame,net,transform):
    height,width=frame.shape[:2]
    frame_t=transform(frame)[0]
    x=torch.from_numpy(frame_t).permute(2,0,1)
    x=Variable(x.unsqueeze(0))
    y=net(x)
    detections=y.data
    print(type(detections))
    scale=torch.Tensor([width,height,width,height])
    print(scale)
    for i in range(detections.size(1)):
        j=0
        while detections[0,i,j,0]>=0.6:
            pt=(detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(50,200,0),2)
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[0])),cv2.FONT_HERSHEY_TRIPLEX,3,(200,20,130),2,cv2.LINE_AA)
            j+=1
    return frame
net=build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location=lambda storage,loc:storage))
transform=BaseTransform(net.size,(104/256.0,11/256.0,123/256.0))
inputimg=imageio.get_reader('Traffic.mp4')
fps=inputimg.get_meta_data()['fps']
outputimg=imageio.get_writer('output.mp4',fps=fps)
for i,frame in enumerate(inputimg):
    frame=detect(frame,net.eval(),transform)
    outputimg.append_data(frame)
    print(i)
outputimg.close()


