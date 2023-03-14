from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout,Embedding
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# train_set = os.listdir("training_set")
# test_set = os.listdir("test_set")

train_datagen = ImageDataGenerator(rescale=(1 / 255.), shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True,
                                   vertical_flip=True)

train_set = train_datagen.flow_from_directory("training_set",
                                              target_size=(255, 255), color_mode='rgb', batch_size=16,
                                              class_mode='binary')

test_datagen = ImageDataGenerator(rescale=(1 / 255.))

test_set = test_datagen.flow_from_directory("test_set", target_size=(255, 255), color_mode='rgb',
                                            batch_size=16, class_mode='binary')

print(train_set[500][0].shape)
print(train_set[500][1].shape)
print(train_set.class_indices)



model = Sequential()
model.add(Conv2D(filters=32, padding="same", activation="relu",
                 kernel_size=3, strides=2, input_shape=(255, 255, 3)))

model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=80, padding="same", activation="relu", kernel_size=3))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, padding="same", activation="relu", kernel_size=3))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=32, padding="same", activation="relu", kernel_size=3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))


 ## For svm classification we add parameter called  “Kernel_regularizer” and inside this regularizer, 
 # we have to use l1 or l2 norm, here I am using l2 norm and pass linear as activation function and that’s 
 # what we did in the final output layer above in the model creation section.

model.add(Dense(1, kernel_regularizer=l2(0.01), activation="linear")) 

#During compiling we’ve to use hinge as a loss function.
model.compile(optimizer="adam", loss="hinge", metrics=["accuracy"])

check_pt = ModelCheckpoint(r"D:\cat and dog\model save folder\model_30.h5", monitor="val_loss",
                           mode='min', save_best_only=True)


train_model = model.fit_generator(train_set, validation_data=test_set, epochs=30,
                                  steps_per_epoch=train_set.samples // 16,
                                  validation_steps=test_set.samples // 16, callbacks=[check_pt])
