import streamlit as st
import numpy as np
import pickle

model=pickle.load(open(r"C:\Users\chand\OneDrive\Desktop\Python\Ml\Emp_Salary_Prediction\linear_regression_model.pkl",'rb'))
st.title("Salary prediction App")

st.write("This app predict the salry based on the year of experience")

year_exp=st.number_input("Enter the year of experience",min_value=0.0, max_value=50.0,value=1.0,step=0.5)

if st.button("predict Salary"):
    exp_input=np.array([[year_exp]])
    prediction=model.predict(exp_input)
    
    st.success(f" The predicted salary for {year_exp} year of exp is: ${prediction[0]:,.2f}")
    
st.write("The model was trained using a dataset of salaries and years of experience.")