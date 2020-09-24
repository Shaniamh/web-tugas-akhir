from flask import Flask, render_template, url_for, request, session, redirect, jsonify
from flask_pymongo import PyMongo
from model.DataTrainPerUser import UserTweet
from model.DataTrainPerUser import Result
from DBRepository.DataTrainPerUser import DataTrainPerUser as DataTrain
from bson import Binary, Code
from bson.json_util import dumps
from flask import flash

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['MONGO_DBNAME'] = 'DB_DepressionAnalyze'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/DB_DepressionAnalyze'

mongo = PyMongo(app)
results = []
collection = mongo.db.DataTrainPerUser

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    # return render_template('analyze.html', menu='analyze')

    return render_template('analyze.html', menu='analyze')


@app.route('/report')
def report():
    return render_template('report.html', menu='report')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        existing_username = collection.find_one({'username' : request.form['username']})
        if existing_username is not None:
            for data in collection.find(existing_username):
                results = data
                print("data : {}".format(results))
                # objUserFind = UserTweet()
                # objUserFind = objUserFind.build_from_json(existing_username)
                # objResult = objUserFind.result
                # result = {'NA': objResult['NA'], 'low': objResult['low'], 'moderate': objResult['moderate'],
                #           'high': objResult['high']}
                # print("result : {}".format(result))
            return render_template('result.html', results=results)

        else:
            flash("The username entered is not in our system")
            return redirect(url_for("analyze"))

    return render_template('result.html')



if __name__ == '__main__':
    app.run(debug=True)

