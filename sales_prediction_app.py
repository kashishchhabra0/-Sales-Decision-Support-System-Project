import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Streamlit app title
st.title("Sales Prediction App")

# User input for predictions
units_sold = st.number_input("Units Sold", min_value=1, value=1)
unit_price = st.number_input("Unit Price", min_value=1.0, value=1.0)
unit_cost = st.number_input("Unit Cost", min_value=1.0, value=1.0)

# Predicting the Total Revenue
if st.button("Predict Total Revenue"):
    input_data = [[units_sold, unit_price, unit_cost]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Total Revenue: ${prediction[0]:,.2f}")
