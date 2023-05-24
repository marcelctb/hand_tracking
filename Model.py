import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing import image

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.05)

training_set = train_datagen.flow_from_directory(
        'Data/com_marcas/training',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'Data/com_marcas/test',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile('adam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

hist = model.fit(training_set, epochs=20,  validation_data=test_set, callbacks=[EarlyStopping(monitor='val_loss', mode='min')])

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate(x=test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

classes = 3
letras = {'0' : 'A', '1' : 'B', '2' : 'C'}

test_image = image.image_utils.load_img('Data/com_marcas/test/C/Image_1684435861.3182466.jpg', target_size=(64, 64))
test_image = image.image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

maior, class_index = -1, -1

for x in range(classes):
   if result[0][x] > maior:
       maior = result[0][x]
       class_index = x

print(result, letras[str(class_index)])

model.save(os.path.join('Model', 'imageclassifier.h5'))