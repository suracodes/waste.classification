import os
import numpy as np
import tensorflow as tf
from flask import Flask, request

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"  # TODO: have a proper secret key for production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_model():
    global model, class_names

    model = tf.keras.models.load_model('weights/mobilenetv3small_acc8652.h5')
    with open('classes.txt', 'r') as f:
        class_names = list(map(lambda x: x.replace('\n', ''), f.readlines()))

    print('Model loaded')


def load_img(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    return img_array


@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['img']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(file.filename))
    file.save(file_path)

    img_array = load_img(file_path)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  # TODO: Remove if confidence is not necessary

    return {"label": class_names[np.argmax(score)],
            "confidence": round(100 * np.max(score), 2)}


@app.route('/', methods=['GET'])
def home():
    return {"msg": "hello, world"}


if __name__ == '__main__':
    load_model()
    app.run(port=5002)
