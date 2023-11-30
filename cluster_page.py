import numpy as np
import pandas as pd
import streamlit as st
import joblib
import time

model = joblib.load('model-kmeans.joblib')
columns = ['total_amount', 'total_transaction']

def pay_stat(x):
  if x > 559579.0:
    return 'High Payment'
  elif x >= 311308.0 and x <= 559579.0:
    return 'Mid Payment'
  else:
    return 'Low Payment'
  
def order_stat(x):
  if x > 74.0:
    return 'High Order'
  elif x >= 39.0 and x <= 74.0:
    return 'Mid Order'
  else:
    return 'Low Order'

def show_cluster_page():

    st.title('Customer Segmentation Prediction')
    st.write("""### Input Customer Data Below""")

    customer_id = st.text_input("Input Customer ID", '123456')

    full_name = st.text_input("Input Customer Fullname", 'Max Verstappen')

    total_amount = st.number_input("Input Total Amount", min_value=1000)

    if total_amount:
      spent_status = pay_stat(total_amount)
      st.code(language='markdown', body=f'Spent Status: {spent_status}')

    total_transaction = st.number_input("Input Total Transaction", min_value=1)

    if total_transaction:
      order_status = order_stat(total_transaction)
      st.code(language='markdown', body=f'Order Status: {order_status}')


    data = pd.DataFrame([[total_amount, total_transaction]], columns=columns)
    
    def predict():
        prediction = model.predict(data)
        
        
        if prediction == 0:
           result_text = "Predicted Cluster: Bronze Member"
        elif prediction == 1: 
             result_text = "Predicted Cluster: Gold Member"
        elif prediction == 2:
            result_text = "Predicted Cluster: Platinum Member"

        result_container.info(result_text)
        time.sleep(3)
        result_container.empty()

    st.button("Predict", on_click=predict)
    result_container = st.empty()



