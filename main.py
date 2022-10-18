import pickle
from sklearn.linear_model import  LogisticRegression
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/AboutUs')
def aboutus():
    return render_template('AboutUs.html')


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        spl = request.form['spl']
        spw = request.form['spw']
        ptl = request.form['ptl']
        ptw = request.form['ptw']

        data: list[list[float]] = [[float(spl), float(spw), float(ptl), float(ptw)]]

        log_model = pickle.load(open('iris.pkl', 'rb'))
        prediction: object = log_model.predict(data)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
