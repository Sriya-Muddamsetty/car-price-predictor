import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

pipe = pickle.load(open('LinearRegressionModel.pkl','rb'))
df = pd.read_csv('cleaned_car_data.csv')

st.title("Car Prediction")

company=st.selectbox('Select Company Name',sorted(df['company'].unique()))
model=st.selectbox('Select Car Model',sorted(df['name'].unique()))
year=st.selectbox('Select Year of Purchase',sorted(df['year'].unique()))
kms_driven=st.number_input('Enter number of kilometers driven')
fuel_type=st.selectbox('Select Fuel type',df['fuel_type'].unique())


if st.button('Predict Price'):
    input = pd.DataFrame([[model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type'])
    st.success("The Predicted car price is ₹"+ str(int(pipe.predict(input)[0])))
