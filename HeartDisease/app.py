import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "knn_HeartDisease.pkl")
scaler = joblib.load(BASE_DIR / "sccaler_HeartDisease.pkl")
Expected_columns = joblib.load(BASE_DIR / "column_Heart.pkl")
st.title("HEART STROKE PREDICTION")
st.markdown("provide the following details")

age=st.slider("age",18,100,40)
sex=st.selectbox("sex",['m','f'])
cheastpain=st.selectbox('ChestPainType',['ATA', 'NAP', 'ASY', 'TA'])
restingBP=st.slider("RestingBP",80,180,120)
Cholesterol=st.slider("Cholesterol",100,600,150)
FastingBS=st.selectbox("FastingBS",['0','1'])
RestingECG=st.selectbox("RestingECG",['Normal', 'ST', 'LVH'])
maxHR=st.slider("MaxHR",71,220,100)
ExerciseAngina=st.selectbox("ExerciseAngina",['yes','no'])
Oldpeak=st.slider('Oldpeak',0.0,5.5,2.2)
st_slope=st.selectbox("ST_slope",['Up', 'Flat', 'Down'])


if st.button("predict"):
    raw_input ={
        "Age":age,
    	"Sex"+sex:1,
    	"ChestPainType"+cheastpain:1,
        "RestingBP":restingBP,
        "Cholesterol":Cholesterol,
        "FastingBS":FastingBS,
        "RestingECG"+RestingECG:1,
        "MaxHR":maxHR,
        "ExerciseAngina"+ExerciseAngina:1,
        "Oldpeak":Oldpeak,
        "ST_Slope"+st_slope:1	
    }


    input_df=pd.DataFrame([raw_input])


    for col in Expected_columns:
        if col not in input_df.columns:
           input_df[col]=0


    input_df = input_df[Expected_columns]     

    scaled_input  =scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]


    if prediction ==1:
        st.error("HIGH HEART RISK")
    else:
        st.success("LOW HEART RISK")    
