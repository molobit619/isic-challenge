from flask import Flask, render_template, jsonify, make_response, Response, request
import numpy as np
from pprint import pprint
import io
import base64
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.math import argmax 

app = Flask(__name__)

trained_models = {
    'model2': {
        'obj': tf.keras.models.load_model(os.path.join('.', 'Resources', 'cnn_model.h5')),
        'dim': (480, 360),
        'reshape': (1, 360, 480, 1),
        'decode': {0: 'benign', 1: 'malignant'},
        'img_read': lambda img_arr: cv.imdecode(img_arr, flags=cv.IMREAD_GRAYSCALE),
        'resize_img': lambda img, dim: cv.resize(img, dim) / 255
    },
    'model1': {
        'obj': tf.keras.models.load_model(os.path.join('.', 'Resources', 'em_model-rgb-adam-weighted.h5')),
        'dim': (120, 187),
        'reshape': (1, 120, 187, 3),
        'decode': {0: 'benign', 1: 'malignant'},
        'img_read': lambda img_arr: cv.imdecode(img_arr, flags=cv.IMREAD_COLOR),
        'resize_img': lambda img, dim: cv.resize(img, dim)
    }
}

@app.route("/process-image", methods=['POST'])
def predict_image():
    if request.method == 'POST': 
        # first 22 characters general format [data:image/jpeg;base64]
        file_payload = request.get_data()[22:].decode()
        file_token = request.get_data()[:22].decode();

        im_bytes = base64.b64decode(file_payload)
        img_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        
        model_details = trained_models['model1'];
        model = model_details['obj']

        img = model_details['img_read'](img_arr)

        resized_img = model_details['resize_img'](img, model_details['dim'])

        prediction = model.predict(
           resized_img.reshape(
                model_details['reshape'] 
           )
        )

        # decode prediction
        argmax_val = argmax(prediction, axis=1)  
        text_prediction = model_details['decode'][int(argmax_val[0])]

        return make_response(jsonify({
                'file_token': file_token,
                'img_shape': img.shape,
                'resize_shape': resized_img.shape,
                'prediction': str(list(prediction[0])),
                'text_prediction': text_prediction
            }))
    else:
        return make_response(jsonify({'message': 'request not supported'}))

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
