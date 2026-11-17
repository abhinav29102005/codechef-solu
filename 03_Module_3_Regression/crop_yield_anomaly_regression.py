import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class CropYieldAnomalyRegression:
    def __init__(self):
        self.dataset=pd.read_csv('crop_yield.csv')
self.alphas=[0.001,0.01,0.1,1,10,100]
self.evaluation_results={}
self.continuous_features=['Area','Annual_Rainfall','Fertilizer','Pesticide']

    def clean_dataset(self):
        numeric=['Crop_Year','Area','Production','Annual_Rainfall','Fertilizer','Pesticide','Yield']
for col in numeric:
    self.dataset[col]=pd.to_numeric(self.dataset[col],errors='coerce')
self.dataset=self.dataset.drop_duplicates().dropna(subset=['Crop_Year','State','Crop','Yield']).copy()
years=np.sort(self.dataset['Crop_Year'].unique())
cut=max(1,min(len(years)-1,int(len(years)*0.8)))
historical=self.dataset[self.dataset['Crop_Year'].isin(years[:cut])]
for col in numeric:
    self.dataset[col]=self.dataset[col].fillna(historical[col].median())
for col in ['State','Crop','Season']:
    mode=historical[col].mode()
    self.dataset[col]=self.dataset[col].fillna(mode.iloc[0] if len(mode) else 'Unknown')

    def split_temporal_data(self):
        self.dataset=self.dataset.sort_values('Crop_Year',kind='stable').copy()
years=np.sort(self.dataset['Crop_Year'].unique())
cut=max(1,min(len(years)-1,int(len(years)*0.8)))
self.train_data=self.dataset[self.dataset['Crop_Year'].isin(years[:cut])].copy()
self.test_data=self.dataset[self.dataset['Crop_Year'].isin(years[cut:])].copy()

    def create_yield_anomaly(self):
        if self.train_data['Crop_Year'].nunique()>=2:
    self.global_trend=np.polyfit(self.train_data['Crop_Year'],self.train_data['Yield'],1)
else:
    self.global_trend=np.array([0.0,self.train_data['Yield'].mean()])
self.trend_models={}
for key,group in self.train_data.groupby(['State','Crop']):
    if len(group)>=2 and group['Crop_Year'].nunique()>=2:
        self.trend_models[key]=np.polyfit(group['Crop_Year'],group['Yield'],1)
for frame in [self.train_data,self.test_data]:
    expected=pd.Series(index=frame.index,dtype=float)
    for key,group in frame.groupby(['State','Crop']):
        trend=self.trend_models.get(key,self.global_trend)
        expected.loc[group.index]=np.polyval(trend,group['Crop_Year'])
    frame['Expected_Yield']=expected
    frame['Yield_Anomaly']=frame['Yield']-expected
self.dataset=pd.concat([self.train_data,self.test_data]).sort_values('Crop_Year',kind='stable')

    def prepare_features(self):
        columns=self.continuous_features+['State','Crop','Season']
train=self.train_data[columns].copy()
test=self.test_data[columns].copy()
for col in ['State','Crop','Season']:
    categories=pd.Index(train[col].unique())
    train[col]=pd.Categorical(train[col],categories=categories)
    test[col]=pd.Categorical(test[col],categories=categories)
encoded=pd.get_dummies(train,columns=['State','Crop','Season'],drop_first=True,dtype=float).astype(float)
encoded_test=pd.get_dummies(test,columns=['State','Crop','Season'],drop_first=True,dtype=float).reindex(columns=encoded.columns,fill_value=0).astype(float)
self.feature_names=encoded.columns.tolist()
self.X_train=encoded.to_numpy(dtype=float)
self.X_test=encoded_test.to_numpy(dtype=float)
self.y_train=self.train_data['Yield_Anomaly'].to_numpy(dtype=float)
self.y_test=self.test_data['Yield_Anomaly'].to_numpy(dtype=float)

    def scale_features(self):
        self.scaler=StandardScaler()
self.X_train=self.scaler.fit_transform(self.X_train)
self.X_test=self.scaler.transform(self.X_test)
self.X_train_scaled=self.X_train
self.X_test_scaled=self.X_test

    def train_linear_model(self):
        self.linear_model=LinearRegression().fit(self.X_train_scaled,self.y_train)

    def train_polynomial_model(self):
        self.poly=PolynomialFeatures(degree=2,include_bias=False)
n=len(self.continuous_features)
self.X_train_poly=np.column_stack([self.poly.fit_transform(self.X_train_scaled[:,:n]),self.X_train[:,n:]])
self.X_test_poly=np.column_stack([self.poly.transform(self.X_test_scaled[:,:n]),self.X_test[:,n:]])
self.polynomial_model=LinearRegression().fit(self.X_train_poly,self.y_train)
self.polynomial=self.poly

    def train_ridge_model(self):
        from sklearn.model_selection import TimeSeriesSplit
self.ridge_model=RidgeCV(alphas=self.alphas,cv=TimeSeriesSplit(n_splits=3),scoring='neg_mean_squared_error').fit(self.X_train_scaled,self.y_train)

    def train_lasso_model(self):
        from sklearn.model_selection import TimeSeriesSplit
self.lasso_model=LassoCV(alphas=self.alphas,cv=TimeSeriesSplit(n_splits=3),max_iter=20000,random_state=42).fit(self.X_train_scaled,self.y_train)

    def evaluate_models(self):
        self.evaluation_results={}
models=[('Linear Regression',self.linear_model,self.X_test_scaled),('Polynomial Regression',self.polynomial_model,self.X_test_poly),('Ridge Regression',self.ridge_model,self.X_test_scaled),('LASSO Regression',self.lasso_model,self.X_test_scaled)]
for name,model,features in models:
    prediction=model.predict(features)
    correlation=float(np.corrcoef(self.y_test,prediction)[0,1]) if np.std(prediction)>0 and np.std(self.y_test)>0 else 0.0
    self.evaluation_results[name]={'rmse':float(np.sqrt(mean_squared_error(self.y_test,prediction))),'mae':float(mean_absolute_error(self.y_test,prediction)),'r2_score':float(r2_score(self.y_test,prediction)),'correlation':correlation}
return self.evaluation_results

    def get_best_model(self):
        if not self.evaluation_results:
    self.evaluation_results=self.evaluate_models()
best=max(v['r2_score'] for v in self.evaluation_results.values())
candidates=[k for k,v in self.evaluation_results.items() if np.isclose(v['r2_score'],best)]
return min(candidates,key=lambda k:self.evaluation_results[k]['rmse'])

    def get_selected_alphas(self):
        return {'Ridge Regression':float(self.ridge_model.alpha_),'LASSO Regression':float(self.lasso_model.alpha_)}

    def get_dataset(self):
        return self.dataset
