import numpy as np
import cv2
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class2text = {
    0: 'mouse',
    1: 'gamepad',
    2: 'shoe',
    3: 'watch'
}
model_filename = "mydataset_item.h5"
model = tf.keras.models.load_model(model_filename)
num_classes = len(class2text)
th = 0.85
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while(cap.isOpened()):
    ret, img = cap.read()
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x = 200
    y = 100
    w = 250
    h = 250

    imgcut = img.copy()[y:y+h, x:x+w]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

    imgcut = cv2.resize(imgcut, (224, 224))
    img_test = np.asarray([imgcut])

    img_test = preprocess_input(img_test)

    probs = model.predict(img_test)
    indexmax = np.argmax(probs[0])
    valmax = np.max(probs[0])

    if (valmax > th):
        name = class2text[indexmax] + " " + str(int(valmax*100)) + "%"
        cv2.putText(img, name, (x, y-5), font, 1, (255, 255, 0), 3)
    else:
        name = "Unknow"
        cv2.putText(img, name, (x, y-5), font, 1, (0, 0, 255), 3)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
