from flask import Flask, render_template, request
import pickle

app=Flask(__name__)
#load the model
model=pickle.load(open('saved_model.sav','rb'))

@app.route('/')
def home():
    result=''
    return render_template('index.html',**locals())

@app.route('/predict', methods=["GET","POST"])
def predict():
     sepal_length1=float(request.form.get('sepal_length',False))
     sepal_width1=float(request.form.get('sepal_width',False))
     petal_length1=float(request.form.get('petal_length',False))
     petal_width1=float(request.form.get('petal_width',False))
     
     result=model.predict([[sepal_length1,sepal_width1,petal_length1,petal_width1]])[0]
     return render_template('index.html',**locals())
   

if __name__=='__main__':
    app.run(debug=True)