import pickle

# Load your saved model (either baseline or enhanced will work for this)
with open('models/enhanced_xgb_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)

# Extract the feature names
try:
    # Works if you used XGBClassifier with a Pandas DataFrame
    expected_features = list(saved_model.feature_names_in_)
except AttributeError:
    # Fallback method
    expected_features = saved_model.get_booster().feature_names

print("⚠️ YOUR STREAMLIT APP MUST USE THESE EXACT COLUMNS IN THIS EXACT ORDER:")
print("-" * 70)
for i, feature in enumerate(expected_features):
    print(f"{i+1}. '{feature}'")
print("-" * 70)