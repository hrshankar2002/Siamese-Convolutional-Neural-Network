import argparse
import os

import cv2
import numpy as np
from alexnet_model import build_alexnet_model
from siamese_model import build_siamese_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model


def load_data(data_dir, target_size=(227, 227)):
    classes = os.listdir(data_dir)
    images_list, labels_list = [], []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images_list.append(img)
            labels_list.append(class_idx)

    images = np.array(images_list)
    labels = np.array(labels_list)

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    return images[indices], labels[indices]


def evaluate_model(model, x_test, y_test, is_siamese=False):
    if is_siamese:
        y_pred = model.predict([x_test, x_test])
    else:
        y_pred = model.predict(x_test)

    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    f2 = fbeta_score(y_test, y_pred_classes, average='weighted', beta=2)

    print(f"Accuracy:  {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"F1 Score:  {f1:.5f}")
    print(f"F2 Score:  {f2:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["alexnet", "siamese"], required=True, help="Model type")
    parser.add_argument("--weights", required=True, help="Path to model weights (.h5)")
    parser.add_argument("--data_dir", type=str, default="data/Dataset", help="Path to dataset")

    args = parser.parse_args()
    print(f"Loading test data from {args.data_dir}...")
    images, labels = load_data(args.data_dir)

    # Split into 80-10-10 like before and use the final 10% as test
    test_split = int(0.9 * len(images))
    x_test = images[test_split:]
    y_test = labels[test_split:]

    if args.model == "alexnet":
        model = build_alexnet_model(input_shape=(227, 227, 3), num_classes=len(set(labels)))
    else:
        model = build_siamese_model(input_shape=(227, 227, 3), num_classes=len(set(labels)))

    print(f"Loading weights from {args.weights}...")
    model.load_weights(args.weights)

    print(f"Evaluating {args.model} model...")
    evaluate_model(model, x_test, y_test, is_siamese=(args.model == "siamese"))


if __name__ == "__main__":
    main()
