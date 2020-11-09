from flask import Flask, render_template, redirect, url_for, request, session, json
import os

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    realImages=os.listdir('static/images/real_images')
    fakeImages=os.listdir('static/images/fake_images')
    return render_template("index.html", realImages=json.dumps(realImages), fakeImages=json.dumps(fakeImages))

@app.route("http://folk.ntnu.no/andeslar/webpage/templates/dcgan")
def dcgan():
    return render_template("DCGAN.html")

@app.route("http://folk.ntnu.no/andeslar/webpage/templates/about")
def bojane():
    return render_template("bojane.html")

if __name__ == "__main__":
    app.run(debug=True)
