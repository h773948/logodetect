from tensorflow.keras.preprocessing.image import (
    img_to_array,
    load_img,
)
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import cv2
import random

HEIGHT = 224
WIDTH = 224
NUMBER_OF_CLASSES = 27
DETECTION_TRESHOLD = 0.95
COUNT_OF_PREDICTIONS_TO_DRAW = 1
SLIDING_WINDOW_CONFIGS = ((120, 40),)
BRAND_LABELS = [
    "Adidas",
    "Apple",
    "BMW",
    "Citroen",
    "Cocacola",
    "DHL",
    "Fedex",
    "Ferrari",
    "Ford",
    "Google",
    "HP",
    "Heineken",
    "Intel",
    "McDonalds",
    "Mini",
    "Nbc",
    "Nike",
    "Pepsi",
    "Porsche",
    "Puma",
    "RedBull",
    "Sprite",
    "Starbucks",
    "Texaco",
    "Unicef",
    "Vodafone",
    "Yahoo",
]

xception = Xception(include_top=False, input_tensor=Input(shape=(WIDTH, HEIGHT, 3)))

head_model = xception.output
head_model = AveragePooling2D(pool_size=(5, 5))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(NUMBER_OF_CLASSES, activation="softmax")(head_model)

model = Model(inputs=xception.input, outputs=head_model)

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.load_weights("./weights.hdf5")


def predict(img):
    test = cv2.resize(img, (WIDTH, HEIGHT))
    test = np.expand_dims(test, axis=0)
    test /= 255
    result = model.predict(test, verbose=0)
    return result


def sliding_window(image, stride, window_size):
    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            yield (x, y, image[y : y + window_size, x : x + window_size].copy())


# configs: List of (int, int) tuples, the first one is the size of the sliding window, the second is the stride
def cnn_detector(img_url, configs):
    img = load_img(img_url)
    img = img_to_array(img)
    significant_preds_for_configs = list()

    for (window_size, stride) in configs:
        significant_preds = dict()

        for (x, y, window) in sliding_window(img, stride, window_size):
            if window.shape[0] != window_size or window.shape[1] != window_size:
                continue

            res = predict(window)
            max_pred = np.amax(res[0])

            if max_pred > DETECTION_TRESHOLD:
                key = BRAND_LABELS[np.argmax(res[0])]
                val = significant_preds.get(key)
                if val:
                    val["x1"] = min(val["x1"], x)
                    val["x2"] = max(val["x2"], x + window_size)
                    val["y1"] = min(val["y1"], y)
                    val["y2"] = max(val["y2"], y + window_size)
                    val["matches"] += 1
                    val["significance"] += max_pred
                    val["significance"] /= 2

                else:
                    significant_preds[key] = {
                        "x1": x,
                        "x2": x + window_size,
                        "y1": y,
                        "y2": y + window_size,
                        "matches": 1,
                        "significance": max_pred,
                    }

        significant_preds_for_configs.append(significant_preds)

    best_preds = dict()

    for dic in significant_preds_for_configs:
        for (brand, params) in dic.items():
            if best_preds.get(brand) is None:
                best_preds[brand] = {
                    "x1": params["x1"],
                    "x2": params["x2"],
                    "y1": params["y1"],
                    "y2": params["y2"],
                    "matches": params["matches"],
                    "significance": params["significance"],
                }

    for (brand, coords) in sorted(
        list(best_preds.items()),
        key=lambda item: item[1]["matches"] * item[1]["significance"],
        reverse=True,
    )[:COUNT_OF_PREDICTIONS_TO_DRAW]:
        (r, g, b) = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        cv2.rectangle(
            img,
            (int(coords["x1"]), int(coords["y1"])),
            (int(coords["x2"]), int(coords["y2"])),
            (r, g, b),
            1,
        )
        cv2.putText(
            img,
            brand,
            (int(coords["x1"]), int(coords["y1"])),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (r, g, b),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite("./output.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    cnn_detector("./test_images/COCACOLA_120_40.jpg", SLIDING_WINDOW_CONFIGS)
