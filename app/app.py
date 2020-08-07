import pickle
import streamlit as st
import numpy as np
import pandas as pd


modelpath = "./model/boston_price.pkl"

@st.cache(suppress_st_warning=True)
def load_model():
    with open(modelpath, "rb") as f:
        model  = pickle.load(f)

    return model

def accept_user_data():
    """
    - CRIM per capita crime rate by town
    - ZN proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS proportion of non-retail business acres per town (small coeff, high p val)
    CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX nitric oxides concentration (parts per 10 million)
    - RM average number of rooms per dwelling
    AGE proportion of owner-occupied units built prior to 1940 (small coeff, high p val)
    - DIS weighted distances to five Boston employment centres
    - RAD index of accessibility to radial highways
    TAX full-value property-tax rate per $10,000
    - PTRATIO pupil-teacher ratio by town
    B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT % lower status of the population

    """
    CRIM = st.slider("Enter the Crime Rate: ", min_value=0 , max_value=100, value=3, step=1)
    ZN = st.slider("Enter the Redidential Proportion of Town: ", min_value=0, max_value=100, value=11, step=1)
    INDUS = 11.1
    CHAS = 0
    NOX = st.slider("Enter the Nitric Oxide Concentration: ", min_value=0, max_value=1, value=0)
    RM = st.slider("Enter Number of Rooms: ", min_value=0, max_value=10, value=6, step=1)
    AGE = 68
    DIS = st.number_input("Enter Average Distance to Employment Centres: ", min_value=0, max_value=None, value=4)
    RAD = st.number_input("Enter Average Distance to a Radial Highway: ", min_value=0, max_value=None, value=9)
    TAX = 408
    PTRATIO = st.number_input("Enter Average Pupil-Teacher Ratio: ", min_value=0, max_value=None, value=19)
    B = 356
    LSTAT = 12

    user_prediction_data = pd.DataFrame(np.array([[CRIM, ZN, INDUS, CHAS, NOX,RM, AGE, DIS,RAD, TAX, PTRATIO, B, LSTAT]]),
                    columns = ['CRIM','ZN', 'INDUS', 'CHAS','NOX','RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    return user_prediction_data


def main():
    st.title("Prediction of Boston Housing Data using Ordinary Least Squares")
    model = load_model()

    user_prediction_data = accept_user_data()
    pred = model.predict(user_prediction_data)

    st.write("The Predicted House Price in Boston is: ", pred)
if __name__ == "__main__":
	main()