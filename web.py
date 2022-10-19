from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
        return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict(final_features).astype(float)
        output=prediction.item()    
        
        out_arr={'Grade 1-Insignificant': 0, 'Grade 2-Low': 1,'Grade 3-Moderate': 2,'Grade 4-Major': 3,'Grade 5-Severe': 4}
        output=list(out_arr.keys())[list(out_arr.values()).index(output)]
        
        return render_template("result.html", prediction_text=output)

if __name__=="__main__":
    app.run(port=5000)
    