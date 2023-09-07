import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='test.jpg')
args = parser.parse_args()

img_path = args.image
model = tf.keras.models.load_model('weights/mobilenetv3small_acc8652.h5')
with open('classes.txt', 'r') as f:
    class_names = list(map(lambda x: x.replace('\n', ''), f.readlines()))

img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

plt.imshow(img)
plt.title(
    "{} with {:.2f}% confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

plt.show()
