import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HousePricePredictionPipeline:
    def __init__(self):
        self.dataset=pd.read_csv('house_price_prediction.csv')

    def clean_dataset(self):
        self.dataset=self.dataset.drop_duplicates().dropna().copy()

    def perform_eda(self):
        return {'total_records':len(self.dataset),'number_of_features':len(self.dataset.columns),'missing_values':int(self.dataset.isna().sum().sum()),'duplicate_rows':int(self.dataset.duplicated().sum())}

    def transform_dataset(self):
        self.X=self.dataset[['HouseSize']]
self.y=self.dataset['HousePrice']

    def train_model(self):
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
self.model=LinearRegression().fit(self.X_train,self.y_train)

    def evaluate_model(self):
        prediction=self.model.predict(self.X_test)
return {'mae':float(mean_absolute_error(self.y_test,prediction)),'mse':float(mean_squared_error(self.y_test,prediction)),'r2_score':float(r2_score(self.y_test,prediction))}

    def predict_price(self, house_size):
        return float(self.model.predict(pd.DataFrame({'HouseSize':[house_size]}))[0])

    def get_dataset(self):
        return self.dataset
