import streamlit as st
import pandas as pd
import plotly.express as px
import time
import joblib
import numpy as np

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Customer Segmentation',
    page_icon='💳',  # This is an emoji shortcode. Could be a URL too.
)

# Load the saved Decision Tree model and scaler
dt_model = joblib.load('./decision_tree_model.pkl')
scaler = joblib.load('./scaler.pkl')

def predict(balance, purchases, credit_limit, resultContainer):
    features = [balance, purchases, credit_limit]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict the tenure category
    predicted_category = dt_model.predict(features_scaled)[0]

    period = "0-7 Months" if predicted_category == "Short-term" else "8-10 Months" if predicted_category == "Mid-term" else "11-24 Months"
    resultContainer.write("Result: The tenure for credit card EMIs should be **:blue[" + predicted_category + "]** period or **:blue[" + period + "]**.")
    time.sleep(2)
    return True

# Load your dataset
displayData = pd.read_csv('./data/display_data.csv')

# ----------------------------------------------------------------------------- 
# Draw the actual page 

# Set the title that appears at the top of the page. 
st.title("💳 Customer Segmentation using :blue[Decision Tree]")

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = {
    "Description": "#description",
    "Prediction": "#prediction",
    "Study of Classification": "#study-classification",
    "Decision Tree Visualization": "#decision-tree"
}
for section, link in sections.items():
    st.sidebar.markdown(f"[{section}]({link})")

# Description Section
st.markdown("<h2 id='description'></h2>", unsafe_allow_html=True)
st.header("Description",divider='gray')
st.write("""
In this project, we are developing a decision tree model to segment customers based on key features from credit card data, specifically *balance*, *purchases*, and *credit limit*. By using these features, we aim to predict customer tenure, which is classified into *short-term*, *mid-term*, and *long-term* categories. This model can assist businesses in understanding customer behavior and tailoring marketing strategies accordingly.

Our approach is inspired by the research conducted in the paper titled ["Study on Application of Customer Segmentation Based on Data Mining Technology"](https://ieeexplore.ieee.org/document/5235679) (DOI: 10.1109/ICCSA.2009.56), which discusses various data mining techniques applied to fraud detection in financial systems. We adapt their methodologies for customer segmentation, focusing on building an efficient decision tree classifier for better customer insights.
""")
''
''

# Prediction Section
st.markdown("<h2 id='prediction'></h2>", unsafe_allow_html=True)
st.header("Prediction",divider='gray')
predictionForm = st.form('predictionForm')
''
balance = predictionForm.number_input("**Balance to be paid**", value=None, placeholder="Type a number...")
purchases = predictionForm.number_input("**Purchases made**", value=None, placeholder="Type a number...")
credit_limit = predictionForm.number_input("**Credit limit on the card**", value=None, placeholder="Type a number...")

if credit_limit is not None and purchases is not None and credit_limit is not None:
    resultContainer = st.container(border=True)
    if predict(balance, purchases, credit_limit, resultContainer):
        predictionForm.form_submit_button('Re-run 🔁')
    else:
        predictionForm.form_submit_button('Done! ✔')
else:
    predictionForm.form_submit_button('Predict 📊')

# Study of Classification Section
st.markdown("<h2 id='study-classification'></h2>", unsafe_allow_html=True)
st.header("Study Classification",divider='gray')

# Drop other columns but keep 'PREDICTED_TENURE_CATEGORY'
chart_data_limited = displayData[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PREDICTED_TENURE_CATEGORY']].head(400)

# Create the 3D scatter plot
fig = px.scatter_3d(chart_data_limited, 
                    x='BALANCE', 
                    y='PURCHASES', 
                    z='CREDIT_LIMIT',
                    color='PREDICTED_TENURE_CATEGORY',
                    title='Credit Card Tenure Clustering',
                    labels={'PREDICTED_TENURE_CATEGORY': 'Predicted Tenure'},
                    color_discrete_map={'Short-term': 'red', 'Mid-term': 'green', 'Long-term': 'blue'},
                    opacity=0.7, 
                    size='CREDIT_LIMIT',  # Point size reflects credit limit
                    size_max=10)  # Increase the max size of points to 10 for visibility

# Update layout for 3D scene titles and set figure size
fig.update_layout(scene=dict(
                    xaxis_title='Balance',
                    yaxis_title='Purchases',
                    zaxis_title='Credit Limit'),
                width=900, height=700)

# Customize marker appearance for better distinction
fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))  # Add outline to points

# Display the plot in Streamlit
st.plotly_chart(fig)

# Decision Tree Visualization Section
st.markdown("<h2 id='decision-tree'></h2>", unsafe_allow_html=True)
st.header("Decision Tree",divider='gray')
st.write("Here is the visualization of the trained Decision Tree model.")
st.write("Entire decision tree")
st.image('decision_tree_from_pkl_matplotlib.png', caption='Entire Decision Tree', use_column_width=True)
st.write("Partail decision tree")
st.image('decision_tree_depth_3.png', caption=' Partial Decision Tree', use_column_width=True)

st.write("""
**Key Components of Decision Trees:**
- **Root Node**: The top node in the tree where the dataset is split first.
- **Decision Nodes**: Nodes that represent decisions based on feature values.
- **Leaf Nodes**: Terminal nodes that represent class labels or regression values.
- **Branches**: Connections between nodes that represent the outcome of a decision.
""")