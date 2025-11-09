import argparse
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def main():
    ap = argparse.ArgumentParser(description="Classify an image with ResNet50 (ImageNet).")
    ap.add_argument("--image", required=True, help="Path to image")
    args = ap.parse_args()

    model = ResNet50(weights="imagenet")

    img = image.load_img(args.image, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    print("\nTop-5 predictions:")
    for _, label, score in decode_predictions(preds, top=5)[0]:
        print(f"  {label:20s} {score:.3f}")

if __name__ == "__main__":
    main()
