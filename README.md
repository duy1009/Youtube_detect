### Cài thư viện
`pip install -r yolov7/requirements.txt`

### Thay đổi đường dẫn
Vào flie `config.py` Sửa `WEIGHT_PATH` và `STREAM_URL` thành đường dẫn của model yolov7 và url livestream của youtube.\
Ví dụ:
```
WEIGHT_PATH = r"E:\ASElab-Data\YOLOV7\xp-11_model\best_2.pt"
STREAM_URL = "https://www.youtube.com/watch?v=Emm2h0i6yAY"
```
### Chạy chương trình

```
python detect_victim.py
```

### Một số lỗi
- Lỗi thư viện pytube:
```
pytube.exceptions.RegexMatchError: get_transform_object: could not find match for var for={(.*?)};
```
Tham khảo: https://stackoverflow.com/questions/76704097/pytube-exceptions-regexmatcherror-get-transform-object-could-not-find-match-fo?fbclid=IwAR00r18hwgwoyBs6vlGnt6a4Q_ndRY3pZGXveb_Oar2OFdQB8qFcGtlOx94

