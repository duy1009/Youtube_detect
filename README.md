### Cài thư viện
`pip install -r yolov7\requirements.txt`

### Detect images/video/rtsp url

```
cd yolov7
python detect.py --weights [đường dẫn file last.pt] --source [đường dẫn folder ảnh/video/rtsp url] --view-img
```

ví dụ: `python yolov7/detect.py --weights ./last.pt --source ./video.mp4 --view-img`


Kết quả sẽ được lưu vào yolov7/runs/detect
