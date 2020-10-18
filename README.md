# Streamlit as a Model Serving Endpoint

Model is trained via train.py and a pickle file was generated in /model/boston_price.pkl 
To run app to score model in real time via user inputs:
 1. conda create -name streamBoston -file requirements.txt (this installs streamlit and other python package dependencies)
 2. conda activate streamBoston
 2. Run `streamlit run app/app.py`
 
 This would give you a local URL to access, mine was `http://localhost:8501`

Here are some screenshots of the web app:

![Screenshot1](https://github.com/gityow/streamlit_app/tree/master/images/screenshot.png?raw=true)
