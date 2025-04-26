import argparse
import os

import cv2
import numpy as np
from models.alexnet_model import build_alexnet_model
from models.siamese_model import build_siamese_model
from tensorflow.keras.models import save_model
from tqdm import tqdm
from tqdm.keras import TqdmCallback


def load_data(data_dir, target_size=(227, 227)):
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)
    images_list, labels_list = [], []

    print(f"Loading images from '{data_dir}'...")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        img_files = os.listdir(class_dir)
        for img_name in tqdm(img_files, desc=f"Class: {class_name}", leave=False):
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
    return images[indices], labels[indices], num_classes


def split_data(images, labels):
    train_split = int(0.8 * len(images))
    val_split = int(0.1 * len(images))
    x_train, y_train = images[:train_split], labels[:train_split]
    x_val, y_val = images[train_split:train_split + val_split], labels[train_split:train_split + val_split]
    x_test, y_test = images[train_split + val_split:], labels[train_split + val_split:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["alexnet", "siamese"], required=True, help="Model type to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="data/Dataset", help="Path to dataset")
    args = parser.parse_args()

    images, labels, num_classes = load_data(args.data_dir)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(images, labels)

    if args.model == "alexnet":
        model = build_alexnet_model(input_shape=(227, 227, 3), num_classes=num_classes)
        print("Training AlexNet...")
        model.fit(
            x_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val),
            callbacks=[TqdmCallback(verbose=1)]
        )
        save_model(model, "alexnet_model.h5")
        print("✅ AlexNet model saved as 'alexnet_model.h5'")

    elif args.model == "siamese":
        model = build_siamese_model(input_shape=(227, 227, 3), num_classes=num_classes)
        print("Training Siamese model...")
        model.fit(
            [x_train, x_train], y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=([x_val, x_val], y_val),
            callbacks=[TqdmCallback(verbose=1)]
        )
        save_model(model, "siamese_model.h5")
        print("✅ Siamese model saved as 'siamese_model.h5'")


if __name__ == "__main__":
    main()
