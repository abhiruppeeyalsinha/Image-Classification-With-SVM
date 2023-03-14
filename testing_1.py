from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import cv2, os

model_path = r"G:\video tutorial\video tutorial\A.I\cat and dog\model save folder\model_30.h5"
img_path = r"G:\video tutorial\video tutorial\A.I\cat and dog\img folder\cat.JPG"

model = load_model(model_path)

test_img = load_img(img_path, target_size=(255, 255))
disp_img = test_img.copy()
test_img = img_to_array(test_img)
test_img = test_img / 255
test_img = np.expand_dims(test_img, axis=0)
# test_img = test_img.reshape(1, )
result = model.predict(test_img)
result = np.nanmax(result)
print(result)
if (int(result)) == 0:
    plt.imshow(disp_img)
    plt.title("Cat")
    plt.show()
else:
    plt.imshow(disp_img)
    plt.title("Dog")
    plt.show()