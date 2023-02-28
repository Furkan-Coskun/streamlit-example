import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

st.set_page_config(page_title="Group 8 Churn Prediction Project",layout="wide" )


st.markdown(
    """
    <style>

    {
       background: #ffff99; 
       background: -webkit-linear-gradient(to right, #ff0099, #493240); 
       background: linear-gradient(to right, #ff0099, #493240); 
    }

    </style>
    """,
    unsafe_allow_html=True,
)



hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
              background: #b8d2d9; 
              background: -webkit-linear-gradient(to down, #ff0099, #493240); 
              background: linear-gradient(to down, #ff0099, #493240); 
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
#title text
st.markdown("<h1 style='text-align: center; color: purple;'>Group 8</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: purple;'>Churn Prediction</h1>", unsafe_allow_html=True)


st.title(" ")

 
satisfaction_level = st.sidebar.slider("Satisfaction Level", min_value=0.09, max_value=1.0, value=0.66, step=0.01)
last_evaluation = st.sidebar.slider("Last Evaluation", min_value=0.36, max_value=1.0, value=0.72, step=0.01)
number_project = st.sidebar.slider("Number of Projects", min_value=2, max_value=7, value=4, step=1)
average_montly_hours = st.sidebar.slider("Average Monthly Hours", min_value=96, max_value=310, value=200, step=1)
time_spend_company = st.sidebar.slider("Time spend in company", min_value=2, max_value=10, value=3, step=1)
department=st.sidebar.selectbox("Department", ['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'])
salary=st.sidebar.selectbox("Salary", ['Low', 'Medium', 'High'])
Work_accident = st.sidebar.checkbox("Accident")
promotion = st.sidebar.checkbox("Promotion in last five years")

my_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "departments":department,
    "salary":salary,
    "work_accident":Work_accident,
    'promotion_last_5years':promotion
}



pt = pickle.load(open('power_transformer', 'rb'))
column_trans=pickle.load(open('transformer','rb'))
model = pickle.load(open('final_model', 'rb'))

df=pd.DataFrame.from_dict([my_dict])
st.table(df)


if st.button("Predict"):
    X=df.copy()
    X['time_spend_company'] = pt.transform(X[['time_spend_company']])
    X_trans = column_trans.transform(X)
    y_pred = model.predict(X_trans)
    if str(y_pred[0])== 0:
       st.success("Employee status: "+str(y_pred[0]))
    else:
       st.error("Employee status: "+str(y_pred[0]))

 



