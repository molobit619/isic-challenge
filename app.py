from flask import Flask, render_template, jsonify, make_response, Response, request
import numpy as np
from pprint import pprint
import io
import base64
import cv2 as cv
import numpy as np
import os

app = Flask(__name__)

@app.route("/process-image", methods=['POST'])
def predict_image():
    if request.method == 'POST': 
        # first 22 characters general format [data:image/jpeg;base64]
        file_payload = request.get_data()[22:].decode()
        file_token = request.get_data()[:22].decode();

        im_bytes = base64.b64decode(file_payload)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)
        
        #todo
        load_model(None)
        resize_image(img)
        predict_image(img)
        
        return make_response(jsonify({'file_token': file_token, 'img_shape': img.shape}))
    else:
        return make_response(jsonify({'wtf': 'wtf'}))

def resize_image(img):
    pass

def load_model(model):
    pass

def predict_image(img):
    pass

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
