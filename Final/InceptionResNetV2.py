from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model

IMAGE_SIZE = (299, 299)

# Step 1: Load InceptionResNetV2 pre-trained model
net = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

number_of_layers = 0

# set this variable to false to not train existing weights
for layer in net.layers:
    layer.trainable = False
    number_of_layers += 1
    
# Step 2: Number of layers in model
print(number_of_layers)

# Step 2: network summary and last 5 layers
print(net.summary())
# Conv2D, Lambda, Conv2D, BatchNormalization, Activation


x = net.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=net.input, outputs=predictions)

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

images_dir = 'img'
img_predictions = {}
# Step 3: Print predictions for test set
for img in os.listdir(images_dir):
    if img.endswith(".jpeg"):
        img_path = os.path.join(images_dir, img)
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = preprocess_input(test_image)

        pred = model.predict(test_image)
        img_predictions[img_path] = pred
    else:
        continue
    
print(img_predictions)

from keras.metrics import top_k_categorical_accuracy

for p in img_predictions:
    top1 = top_k_categorical_accuracy(y_true, img_predictions[p], k=1)
    top5 = top_k_categorical_accuracy(y_true, img_predictions[p], k=5)

