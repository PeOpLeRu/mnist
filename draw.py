from keras.models import Sequential, load_model
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

NAME_MODEL = "mnist.h5"

draw = False

def draw_callback(event, x, y, flags, param):
    global draw

    if event == cv.EVENT_MOUSEMOVE:
        if draw:
            cv. circle(img, (x, y), 15, 200, -1)

    elif event == cv.EVENT_LBUTTONDOWN:
        draw = True
    elif cv.EVENT_LBUTTONUP:
        draw = False

def predict_image(img):
    blurred_img = cv.GaussianBlur(img, (45, 45), 40)

    resized_img = cv.resize(blurred_img, (28, 28))

    normalized_img = resized_img / 255

    prepared_img = normalized_img.reshape(1, *normalized_img.shape, 1)

    predict = model.predict(prepared_img)

    answer = np.argmax(predict)
    answer_probability = str(np.round(predict[0], 2)[answer])

    return answer, answer_probability

# -------- main code --------

img = np.zeros((512, 512), dtype='uint8')

if os.path.exists(NAME_MODEL):
    model = load_model(NAME_MODEL)
    model.summary()
else:
    raise FileNotFoundError("File with model not found")

cv.namedWindow("MNIST")
cv.setMouseCallback("MNIST", draw_callback)

while True:
    cv.imshow("MNIST", img)

    key = cv.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('m'):
        answer, probability = predict_image(img)
        print(f"Recognized -> {answer} ({probability} %)")
    elif key == ord('c'):
        img[:] = 0

cv.destroyAllWindows()