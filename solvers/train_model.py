import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def generate_printed_digits(samples_per_digit=400):
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]
    images, labels = [], []

    for digit in range(1, 10):
        for _ in range(samples_per_digit):
            img = np.zeros((28, 28), dtype=np.uint8)
            font = np.random.choice(fonts)
            scale = np.random.uniform(0.65, 0.9)
            thickness = np.random.randint(1, 3)

            (w, h), baseline = cv2.getTextSize(str(digit), font, scale, thickness)
            x = (28 - w) // 2 + np.random.randint(-2, 3)
            y = (28 + h) // 2 + np.random.randint(-2, 3)

            cv2.putText(img, str(digit), (x, y), font, scale, 255, thickness, cv2.LINE_AA)
            images.append(img)
            labels.append(digit)

    return np.array(images), np.array(labels)

def train_and_save_model(model_path="digit_model.keras"):
    # Load MNIST and filter out 0s
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_mask = y_train > 0
    test_mask = y_test > 0
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Add synthetic printed digits for high accuracy on printed grids
    syn_x, syn_y = generate_printed_digits()
    x_train = np.concatenate([x_train, syn_x], axis=0)
    y_train = np.concatenate([y_train, syn_y], axis=0)

    # Normalize to [0, 1] and reshape to (28, 28, 1)
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    train_and_save_model()