import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the saved Decision Tree model from the .pkl file
dt_model = joblib.load('decision_tree_model.pkl')

# Define feature names and class names (these should match what you used in the training phase)
feature_names = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']
class_names = ['Short-term', 'Mid-term', 'Long-term']

# Create a plot of the decision tree, limiting to depth 5
plt.figure(figsize=(20,10))
plot_tree(dt_model, 
          feature_names=feature_names, 
          class_names=class_names, 
          filled=True, 
          rounded=True, 
          fontsize=10, 
          max_depth=3)  # Limit to depth 5

# Save the plot as a PNG image
plt.savefig('decision_tree_depth_3.png')

# Optionally, show the plot
plt.show()

print("Decision tree (up to depth 5) saved as 'decision_tree_depth_5.png'.")
