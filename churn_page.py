import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder
import time


model = joblib.load('model-xgboost.joblib')
le = LabelEncoder()
columns = ['customer_gender', 'age', 'age_cat', 'device_type', 'home_location',
          'home_country', 'total_year_join', 'promo_amount', 'total_amount',
          'spent_status', 'total_transaction', 'order_status', 'using_promo',
          'prefered_season_product', 'prefered_cat', 'prefered_payment_method',
          'prefered_time_access', 'year_avg_activity']


def age_category(x):
  if x > 40:
    return 'Orang tua'
  elif x <= 40 and x >= 20:
    return 'Dewasa'
  elif x < 20 and x >= 15:
    return 'Remaja'
  else:
    return 'Anak-Anak'
  
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

def promo_stat(x):
  if x == 0:
    return 'No'
  else:
    return 'Yes'
  
def encode_features(data):
    encoding_dict = {}
    for feature in data.columns:
        if data[feature].dtype == 'object':
            encoding_dict[feature] = le.fit_transform(data[feature])
        else:
            encoding_dict[feature] = data[feature].values[0]
    return encoding_dict


def show_churn_page() :
    st.title('Customer Churn Prediction')
    st.subheader('Input Customer Data Below')

    customer_id = st.text_input("Customer ID", '123456')
    full_name = st.text_input("Customer Fullname", 'Max Verstappen')
    customer_gender = st.selectbox("Customer Gender", ['Male', 'Female'])

    min_date = datetime.date(1900, 1, 1)
    birthdate = st.date_input('Customer Birth Date', min_value=min_date)

    age = st.number_input("Customer Age", min_value=1, max_value=100)
    if age:
       age_cat = age_category(age)
       st.code(language='markdown', body=f'Age Category: {age_cat}')

    device_type  = st.selectbox("Device Type", ['Android', 'Ios'])

    city = (
       'Nusa Tenggara Barat', 'Jakarta Raya', 'Jawa Timur',
       'Kalimantan Barat', 'Jawa Tengah', 'Jawa Barat', 'Lampung',
       'Yogyakarta', 'Bangka Belitung', 'Papua Barat', 'Maluku', 'Bali',
       'Kalimantan Timur', 'Kalimantan Selatan', 'Sulawesi Barat',
       'Kalimantan Tengah', 'Kepulauan Riau', 'Gorontalo',
       'Sumatera Utara', 'Sumatera Barat', 'Sulawesi Selatan',
       'Sumatera Selatan', 'Sulawesi Tengah', 'Papua',
       'Nusa Tenggara Timur', 'Jambi', 'Banten', 'Sulawesi Utara',
       'Bengkulu', 'Aceh', 'Maluku Utara', 'Riau', 'Sulawesi Tenggara')

    home_location = st.selectbox("Home Location", city)

    home_country = st.text_input("Country", "Indonesia")

    total_year_join = st.number_input('Total Year Using App', min_value=1, max_value=100)

    using_promo = None
    promo_amount = st.number_input("Input Promo Amount", min_value=0)


    total_amount = st.number_input("Input Total Amount", min_value=1000)

    if total_amount:
      spent_status = pay_stat(total_amount)
      st.code(language='markdown', body=f'Spent Status: {spent_status}')

    total_transaction = st.number_input("Input Total Transaction", min_value=1)

    if total_transaction:
      order_status = order_stat(total_transaction)
      st.code(language='markdown', body=f'Order Status: {order_status}')

    if promo_amount:
      using_promo = promo_stat(promo_amount)
      st.code(language='markdown', body=f'Customer Using Promo? : {using_promo}')
    
    season = ('Summer', 'Fall', 'Spring', 'Winter', 'other')

    prefered_season_product = st.selectbox("Prefered Season Product", season)

    category = ('Apparel', 'Accessories', 'Footwear', 
                'Personal Care', 'Free Items', 'Sporting Goods')
    prefered_cat = st.selectbox("Prefered Product Category", category)

    payment = ('LinkAja', 'Credit Card', 'Debit Card', 'OVO', 'Gopay')
    prefered_payment_method = st.selectbox("Prefered Payment Method", payment)

    time_access = ('evening', 'morning', 'night', 'afternoon')
    prefered_time_access = st.selectbox("Prefered Time Access", time_access)

    year_avg_activity = st.number_input("Average Activity per Year", min_value=1, max_value=5000)


    def predict():
      data = pd.DataFrame([[customer_gender, age, age_cat, device_type, home_location,
                            home_country, total_year_join, promo_amount, total_amount,
                            spent_status, total_transaction, order_status, using_promo,
                            prefered_season_product, prefered_cat, prefered_payment_method,
                            prefered_time_access, year_avg_activity]], columns=columns)
      
      encoding_dict = encode_features(data)
      x = pd.DataFrame([list(encoding_dict.values())], columns=columns)

      prediction = model.predict(x.values)
      

      if prediction == 1:
        result_text = "Customer is Possibly Churn"
        result_container.error(result_text, icon="📈")
      else: 
        result_text = "Customer is Possibly Not Churn"
        result_container.success(result_text, icon="📉")

      time.sleep(3)
      result_container.empty()  

    st.button("Predict", on_click=predict)  
    result_container = st.empty()
  
      



