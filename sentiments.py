from asyncio.windows_events import NULL
from urllib import response
from flask import Flask,request, render_template, Blueprint, jsonify, json
app = Flask(__name__)
import engine as en
from werkzeug.utils import secure_filename
# import pandas as pd
from flask_cors import CORS
CORS(app)

app.config["UPLOAD_FOLDER"] = "static/"
second = Blueprint("second", __name__, static_folder="static", template_folder="template")


# @app.route('/api/v1/getAnalysis/', methods=['GET','POST'])
@second.route('/sentiment_analyzer', methods=['GET', 'POST'])
def sentiment_analyzer():
    # # handle the POST request
    
    final_result = {}
    final_data = []
    negetive = 0
    neutral = 0
    positive = 0
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(app.config['UPLOAD_FOLDER'] + filename)

        file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        content = file.read()

        print(content) # Till here the form data input is read by the flask code snippest.
        #now we will work with the API.

        final_result = en.final_analysis(content)
        json.dumps(final_result, indent = 3)
        print(final_result)
        final_data = list(final_result.values())
        negetive = final_data[0]
        neutral = final_data[1]
        positive = final_data[2]
        print(negetive)

    return render_template('sentiment_analyzer.html', negetive = negetive, neutral = neutral, positive = positive )  


    



