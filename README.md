# ECE588_tracking
This is a course project of ECE588 at Duke University. In this project, a multi-object tracking algorithm [ByteTrack](https://github.com/ifzhang/ByteTrack) is implemented and used to track people in videos.

## Setup
For setup, please clone the repository and install the required packages, 
```
git clone https://github.com/SiyuZhou918/ECE588_tracking.git
cd ECE588_tracking
pip install -r requirements.txt
python setup.py develop
```

## Data and pretrained weights preparation
Download the [Multiple Object Tracking (MOT)](https://motchallenge.net/data/MOT17/) dataset and pretrained weights of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), and save them in `ECE588_tracking/datasets` and `ECE588_tracking/pretrained` separately. If you are using Ubuntu, you can use the following codes:
```
mkdir datasets
cd datasets
wget -c "https://motchallenge.net/data/MOT17.zip"
unzip MOT17.zip
cd ..

mkdir pretrained
cd pretrained
wget -c "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
wget -c "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
cd ..
```

## Dataset preprocessing
Please run the codes:
```
python tools/convert_mot17_to_coco.py
```

## Training and fine-tuning a YOLOX model
Load pretrained YOLOX-nano and fine-tune in MOT17 dataset:
```
python tools/train.py -f exps/yolox_nano_mot.py -b 8 -c pretrained/yolox_nano.pth
```

YOLOX-tiny:
```
python tools/train.py -f exps/yolox_tiny_mot.py -b 4 -c pretrained/yolox_tiny.pth
```



## Association
* Palace video download: https://drive.google.com/file/d/1Ye5Nw7VhTvZVMiniqZoWvbvoScMkOZ72/view?usp=sharing
* For testing on cpu, download: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth 