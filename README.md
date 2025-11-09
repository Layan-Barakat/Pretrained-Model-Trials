# Pretrained Model Trials

Tiny demos for popular pretrained models:
- **YOLOv8** (Ultralytics) — real-time webcam object detection
- **VGG16** — ImageNet classification
- **ResNet50** — ImageNet classification

---

## Structure
```text
.
├─ scripts/
│  ├─ yolov8_webcam.py
│  ├─ vgg16_classify.py
│  └─ resnet50_classify.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run
```bash
# YOLOv8 webcam detection
python scripts/yolov8_webcam.py

# Classify an image (VGG16)
python scripts/vgg16_classify.py --image path/to/image.jpg

# Classify an image (ResNet50)
python scripts/resnet50_classify.py --image path/to/image.jpg
```

> If you hit an OpenMP/KMP error on some systems, try:
> ```python
> import os
> os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
> ```
> (Place it at the very top of the script, before importing TensorFlow.)

---

## Author
Layan Barakat — University of Birmingham Dubai
