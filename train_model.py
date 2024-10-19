import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score  # Import accuracy_score
import joblib
import plotly.express as px


# Load your dataset
data = pd.read_csv('./data/CreditCard_data.csv')

# Define a function to categorize tenure
def categorize_tenure(tenure):
    if tenure <= 7:
        return 'Short-term'
    elif tenure <= 10:
        return 'Mid-term'
    else:
        return 'Long-term'

# Apply the categorization function
data['TENURE_CATEGORY'] = data['TENURE'].apply(categorize_tenure)

# Select only the necessary columns for training
features = data[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']]
target = data['TENURE_CATEGORY']  # Use the new categorical column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved successfully.")

# Make predictions on the test set
y_pred = dt_model.predict(X_test_scaled)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Optionally, display the tenure and tenure category for a sample of the data
sample_data = pd.DataFrame({
    'BALANCE': X_test['BALANCE'],
    'PURCHASES': X_test['PURCHASES'],
    'CREDIT_LIMIT': X_test['CREDIT_LIMIT'],
    'TENURE': y_test,  # Keeping the original tenure for reference
    'PREDICTED_TENURE_CATEGORY': y_pred
})

# Show the results
print("\nSample Data with Tenure and Predicted Tenure Category:")
print(sample_data.head())  # Display the first few records

# Extract accurately predicted rows for each tenure category
accurate_short_term = sample_data[(sample_data['PREDICTED_TENURE_CATEGORY'] == 'Short-term') & (sample_data['TENURE'] == 'Short-term')]
accurate_mid_term = sample_data[(sample_data['PREDICTED_TENURE_CATEGORY'] == 'Mid-term') & (sample_data['TENURE'] == 'Mid-term')]
accurate_long_term = sample_data[(sample_data['PREDICTED_TENURE_CATEGORY'] == 'Long-term') & (sample_data['TENURE'] == 'Long-term')]

# Get one row for each accurate tenure category
result_short_term = accurate_short_term.head(1) if not accurate_short_term.empty else None
result_mid_term = accurate_mid_term.head(1) if not accurate_mid_term.empty else None
result_long_term = accurate_long_term.head(1) if not accurate_long_term.empty else None

# Combine results into a single DataFrame
results = pd.concat([result_short_term, result_mid_term, result_long_term], ignore_index=True)

# Print the results
print("\nAccurately Predicted Rows for Each Tenure Category:")
print(results)