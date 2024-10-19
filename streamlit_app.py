import streamlit as st
import pandas as pd
import time
import joblib
import numpy as np
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Customer Segmentation',
    page_icon='💳', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Load the saved Decision Tree model and scaler
dt_model = joblib.load('./decision_tree_model.pkl')
scaler = joblib.load('./scaler.pkl')

def predict(balance, purchases, credit_limit,resultContainer):
    features = [balance, purchases, credit_limit]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict the tenure category
    predicted_category = dt_model.predict(features_scaled)[0]

    period="0-7 Months" if predicted_category=="Short-term" else "8-10 Months" if predicted_category=="Mid-term" else "11-24 Months"
    resultContainer.write("Result: The tenure for credit card EMIs should be **:blue["+predicted_category+"]** period or **:blue["+period+"]**.")
    time.sleep(2)
    return True
    
# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.

st.title("💳 Customer Segmentation using :blue[Decision Tree]")

'''

In this project, we are developing a decision tree model to segment customers based on key features from credit card data, specifically *balance*, *purchases*, and *credit limit*. By using these features, we aim to predict customer tenure, which is classified into *short-term*, *mid-term*, and *long-term* categories. This model can assist businesses in understanding customer behavior and tailoring marketing strategies accordingly.

Our approach is inspired by the research conducted in the paper titled ["Study on Application of Customer Segmentation Based on Data Mining Technology"](https://ieeexplore.ieee.org/document/5235679) (DOI: 10.1109/ICCSA.2009.56), which discusses various data mining techniques applied to fraud detection in financial systems. We adapt their methodologies for customer segmentation, focusing on building an efficient decision tree classifier for better customer insights.


'''

# Add some spacing
''
''

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

st.header('Prediction using your input', divider='gray')
''
predictionForm=st.form('predictionForm')

balance = predictionForm.number_input(
    "**Balance to be paid**", value=None, placeholder="Type a number..."
)
purchases = predictionForm.number_input(
    "**Purchases made**", value=None, placeholder="Type a number..."
)
credit_limit = predictionForm.number_input(
    "**Credit limit on the card**", value=None, placeholder="Type a number..."
)
''

if credit_limit != None and purchases != None and credit_limit != None:
    resultContainer=st.container(border=True)
    if predict(balance, purchases, credit_limit, resultContainer):
        predictionForm.form_submit_button('Re-run 🔁')
    else:
        predictionForm.form_submit_button('Done! ✔')

else:
    predictionForm.form_submit_button('Predict 📊')
