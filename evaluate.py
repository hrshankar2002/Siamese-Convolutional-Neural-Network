import argparse
import os

import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from models.alexnet_model import build_alexnet_model
from models.siamese_model import build_siamese_model


def load_data(data_dir, target_size=(227, 227)):
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)
    images_list, labels_list = [], []

    print(f"Loading images from '{data_dir}'...")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        img_files = os.listdir(class_dir)
        for img_name in img_files:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images_list.append(img)
            labels_list.append(class_idx)

    images = np.array(images_list)
    labels = np.array(labels_list)

    return images, labels, num_classes, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["alexnet", "siamese"], required=True)
    parser.add_argument("--weights", required=True, help="Path to saved .h5 model file")
    parser.add_argument("--data_dir", required=True, help="Directory containing evaluation dataset")
    args = parser.parse_args()

    # Load test data
    X, y, num_classes, class_names = load_data(args.data_dir)

    # Load model
    print(f"Loading model from '{args.weights}'...")
    model = load_model(args.weights, compile=True)

    print(f"Evaluating {args.model} model...")

    if args.model == "alexnet":
        y_pred_probs = model.predict(X)
        y_pred = np.argmax(y_pred_probs, axis=1)

    elif args.model == "siamese":
        # Siamese model expects two inputs: image1 and image2
        y_pred_probs = model.predict([X, X])
        y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    main()
