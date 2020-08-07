# Tutorial Here
# https://github.com/ShivamBhirud/Capital-Bike-Share-Data-Streamlit-Web-Application/blob/master/demoStreamlit.py
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

modelpath = "./model/boston_price.pkl"

def load_data():
    data = datasets.load_boston()
    X = pd.DataFrame(data['data'],columns=data['feature_names'])
    Y = pd.DataFrame(data['target'], columns=['MEDV'])
    return X, Y


def linreg(X, Y):
    model = LinearRegression().fit(X, Y)
    #score = model.score(X,Y) # 0.7406426641094095
    with open(modelpath, "wb") as f:
        pickle.dump(model, f)


