import time
import cv2
import torch
from yolov7.ASE_Detect_yolov7 import Detect
from yolov7.utils.general import scale_coords
import streamlink
from threading import Thread
from config import *

if STREAM_URL[:len("https://www.youtube.com")] == "https://www.youtube.com":
    try:
        streams = streamlink.streams(STREAM_URL)
        # Lấy stream video live từ YouTube
        stream_url = streams["best"].url

        print(stream_url)
    except:
        from pytube import YouTube
        # Choose a video stream with resolution of 360p
        streams = YouTube(STREAM_URL).streams.filter(adaptive=True, subtype="mp4", resolution="480p", only_video=True)

        # Check if there is a valid stream
        if len(streams) == 0:
            raise "No suitable stream found for this YouTube video!"

            # Download the video as video.mp4
        print("Downloading...")
        stream_url = streams[0].download(filename="video.mp4")
        print("Download completed.")
else:
    stream_url = STREAM_URL


# Mở stream video từ URL
vid = cv2.VideoCapture(stream_url)

Det = Detect(WEIGHT_PATH, im_size, device)

is_detc = True
stop = False
FRAME = None
PRED = None

def detect():
    global FRAME, stop, PRED, Det
    while True:
        if stop:
            break
        st = time.time()
        if FRAME is None:
            continue
        PRED = Det.detect(FRAME ,conf_thres = conf_thres, iou_thres=iou_thres) # result 
        # Predict value
        # pred_rescale = torch.tensor(pred[0])
        # pred_rescale[:, :4] = scale_coords(Det.img_size_detect[2:], pred_rescale[:, :4], img.shape).round()
        # center = Det.get_center(pred_rescale[:4])[:,:2]
        # _type = pred_rescale[:,5]
        print(f"TPF: {time.time()-st}")
thr = Thread(target=detect, name="thr")
thr.start()

while True:
    with torch.no_grad(): 
        ret, img = vid.read() 
        FRAME = img.copy()  
        if not ret:
            break
        if PRED:
            img_rstl = Det.draw_all_box(img=img,pred=PRED) # draw box
        else:
            img_rstl = img

        cv2.imshow("Test", img_rstl)
        event =  cv2.waitKey(1)
        if  event== ord('q'):
            break

stop = True
vid.release()
cv2.destroyAllWindows()