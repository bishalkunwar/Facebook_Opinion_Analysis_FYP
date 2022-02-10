from flask import Flask,request,jsonify, render_template
app = Flask(__name__)
import engine as en
from flask_cors import CORS
CORS(app)



# @app.route('/api/v1/getAnalysis/', methods=['GET','POST'])
@app.route('/sentiment_analyzer', methods=['GET', 'POST'])
def getMessage():
    # handle the POST request
    
    message = request.get('message') 
    final_result = en.final_analysis(message)
    return render_template('sentiment_analyzer.html', positive = final_result.pos, negetive = final_result.neg, neutral = final_result.neu)


    # content = request.json
    # message = content["message"]
    # print(message)
    # final_result = en.final_analysis(message)
    # return jsonify({"data":final_result})

    