from flask import Flask, render_template, jsonify, make_response, Response
import numpy as np
import io

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
