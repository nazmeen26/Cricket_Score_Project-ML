from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Match_ID=float(request.form.get('Match_ID')),
            Overs_Played=float(request.form.get('Overs_Played')),
            Wickets_Lost=float(request.form.get('Wickets_Lost')),
            Run_Rate=float(request.form.get('Run_Rate')),
            Opponent_Strength=float(request.form.get('Opponent_Strength')),
            Home_Away=request.form.get('Home_Away'),
            Pitch_Condition=request.form.get('Pitch_Condition'),
            Weather=request.form.get('Weather')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        

        predict_pipeline=PredictPipeline()
        
        results=predict_pipeline.predict(pred_df)
        
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        