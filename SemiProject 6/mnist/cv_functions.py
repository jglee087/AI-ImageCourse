from django.conf import settings
import numpy as np
import cv2
from tensorflow.keras import models

def cv_detect_number(path):

    baseUrl = settings.MEDIA_ROOT_URL + settings.MEDIA_URL
    model = models.load_model(baseUrl+'mnist_2layer_bn.h5')

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)

    gray = cv2.resize(img, dsize = (28,28),interpolation=cv2.INTER_AREA)
    img = gray.reshape((1,28*28))
    result = model.predict_classes(img)
    return result
    # #
    # if (type(gray) is np.ndarray) :
    #     img = img.reshape((1,28*28))
    #     result = model.predict_classes(img)
    #     #cv2.imwrite(path, img)
    # else:
    #     print('Error occurred within cv_detect_face!')
    #     print(path)
