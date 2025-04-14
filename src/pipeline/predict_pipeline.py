import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Match_ID: int,
        Overs_Played: int,
        Wickets_Lost: int,
        Run_Rate: int,
        Opponent_Strength: int,
        Home_Away,
        Pitch_Condition: str,
        Weather: str):

        self.Match_ID = Match_ID

        self.Overs_Played = Overs_Played

        self.Wickets_Lost = Wickets_Lost

        self.Run_Rate = Run_Rate

        self.Opponent_Strength = Opponent_Strength

        self.Home_Away= Home_Away

        self.Pitch_Condition = Pitch_Condition

        self.Weather = Weather

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Match_ID": [self.Match_ID],
                "Overs_Played": [self.Overs_Played],
                "Wickets_Lost": [self.Wickets_Lost],
                "Run_Rate": [self.Run_Rate],
                "Opponent_Strength": [self.Opponent_Strength],
                "Home_Away": [self.Home_Away],
                "Pitch_Condition": [self.Pitch_Condition],
                "Weather": [self.Weather],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)