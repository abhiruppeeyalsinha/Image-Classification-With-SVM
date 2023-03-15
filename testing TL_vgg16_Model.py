import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np,cv2
import matplotlib.pyplot as plt


img = cv2.imread('image.jpg')
model_path = r"G:\video tutorial\video tutorial\A.I\cat and dog\model save folder\tl_vgg16_30.h5"
train_model = load_model(model_path)
# print(train_model.summary())
img_path = r"G:\video tutorial\video tutorial\A.I\cat and dog\img folder\dog-and-cat.jpg"

test_img =  load_img(img_path, target_size=(255,255))
dispImg = test_img.copy()
test_img = img_to_array(test_img)
test_img = test_img/255
test_img = np.expand_dims(test_img,0)
model_eva = train_model.evaluate(test_img,verbose=0)
print(model_eva)
res = train_model.predict(test_img)
print(res)
res = int(np.nanmax(res))
# print(res)
if int(res <= 0 ):
    plt.imshow(dispImg)
    plt.title('Cat')
    plt.show()
elif int(res ==1):
    plt.imshow(dispImg)
    plt.title('Dog')
    plt.show()
else:
    print("Not Listed!!")




