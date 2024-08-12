#the web app
from flask import Flask,render_template,request
import pickle
import numpy as np

model_path='star.pkl'
with open(model_path,'rb') as file:
    model=pickle.load(file)

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    temp=request.form['temp']
    lum=request.form['lum']
    rad=request.form['rad']
    mg=request.form['mg']
    arr=np.array([[temp,lum,rad,mg]])
    predicted=model.predict(arr)
    return render_template('result.html',data=predicted)

if __name__=="__main__":
    app.run(debug=True)

    