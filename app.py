from flask import Flask, render_template, jsonify, make_response, Response, request
import numpy as np
from pprint import pprint
import io
import base64
import cv
import tempfile

app = Flask(__name__)

@app.route("/process-image", methods=['POST'])
def predict_image():
    if request.method == 'POST': 
        #pprint(base64.decodestring(request.form))
        pprint( base64.decodestring(request.get_data()[22:]) )
        with tempfile.TemporaryFile() as fp:
            fp.write( base64.decodestring(request.get_data()[22:]) )
            fp.seek(0)
            img = cv.imread(fp)
        return make_response(jsonify({'wtf': 'is post request'}))
    else:
        return make_response(jsonify({'wtf': 'wtf'}))

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
