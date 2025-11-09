# YOLOv8 webcam demo
# Tip: if you get an OpenMP/KMP error on some machines, set:
#   import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # auto-downloads on first run
    # 0 = default webcam; pass a video path instead to run on video
    model.predict(source=0, show=True)

if __name__ == "__main__":
    main()
