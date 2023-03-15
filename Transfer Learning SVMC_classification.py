from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, GlobalMaxPooling2D, Flatten, Dense, Dropout, Embedding
from keras.preprocessing.image import ImageDataGenerator,image_dataset_from_directory
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# train_set = os.listdir("training_set")
# test_set = os.listdir("test_set")
train_set = r"G:\video tutorial\video tutorial\A.I\cat and dog\training_set"
test_set  =r"G:\video tutorial\video tutorial\A.I\cat and dog\test_set"


train_datagen = ImageDataGenerator(rescale=(1 / 255.), shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True,
                                   vertical_flip=True)

train_set = train_datagen.flow_from_directory(train_set,
                                              target_size=(255, 255), color_mode='rgb', batch_size=16,
                                              class_mode='binary',seed=42,subset='training')

test_datagen = ImageDataGenerator(rescale=(1 / 255.))

test_set = test_datagen.flow_from_directory(test_set, target_size=(255, 255), color_mode='rgb',
                                            batch_size=16, class_mode='binary',seed=42)


base_model =  VGG16(weights='imagenet', pooling=GlobalMaxPooling2D,
                        include_top=False,input_shape = (255,255,3))

base_model.trainable = False


# print(base_model.summary())
model  = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1,kernel_regularizer=l2(0.01),activation='linear'))
# print(model.summary())
model.compile(optimizer='adam',loss='hinge',metrics=['accuracy'])

callback = EarlyStopping(
    monitor='val_loss', patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

train_model =  model.fit_generator(train_set,validation_data=test_set,callbacks=[callback],epochs=30)


train_model.model.save(r"G:\video tutorial\video tutorial\A.I\cat and dog\model save folder\tl_vgg16_30.h5")
