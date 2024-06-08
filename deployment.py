

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: ')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Type',[0,1,2])
    TWF = st.sidebar.selectbox('TWF',[0,1])
    HDF = st.sidebar.selectbox('HDF',[0,1])
    PWF = st.sidebar.selectbox('PWF',[0,1])
    OSF = st.sidebar.selectbox('OSF',[0,1])
    RNF = st.sidebar.selectbox('RNF',[0,1])
    CLMINSUR = st.sidebar.number_input('air_Temp')
    CLMAGE = st.sidebar.number_input("Insert the process_Temp")
    R_speed_rpm = st.sidebar.number_input("Insert R_speed_rpm")
    SEATBELT = st.sidebar.number_input('torque_nm')
    tool_wear = st.sidebar.number_input('tool_wear')
    data = {'Type':CLMSEX,
            'air_Temp':CLMINSUR,
            'process_Temp':CLMAGE,
            'R_speed_rpm':R_speed_rpm,
            'torque_nm':SEATBELT,
            'tool_wear':tool_wear,
            'TWF':TWF,
            'HDF':HDF,
            'PWF':PWF,
            'OSF':OSF,
            'RNF':RNF

            }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('Model.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction)

