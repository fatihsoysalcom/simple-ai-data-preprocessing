import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Simulate raw AI data
data = {
    'feature1': [1.5, 2.3, 0.8, 3.1, 1.9, 2.5, 0.5, 3.8, 1.2, 2.9],
    'feature2': [10, 15, 12, 18, 13, 16, 11, 20, 14, 17],
    'categorical_feature': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical features
numerical_features = ['feature1', 'feature2']
categorical_features = ['categorical_feature']

# Create preprocessing pipelines for numerical and categorical features
# StandardScaler for numerical features (handles scaling)
# OneHotEncoder for categorical features (handles categorical encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a full pipeline that first preprocesses and then can be used for modeling
# For this example, we'll just demonstrate the preprocessing step
# A full AI stack would include a model here, e.g., a classifier

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data using the fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\n--- Processed Training Data (first 5 rows) ---")
# Convert to DataFrame for better readability, showing only first few columns if many from OneHotEncoder
processed_train_df = pd.DataFrame(X_train_processed[:, :len(numerical_features) + len(preprocessor.named_transformers_['cat'].categories_[0])])
print(processed_train_df.head())

print("\n--- Processed Test Data (first 5 rows) ---")
processed_test_df = pd.DataFrame(X_test_processed[:, :len(numerical_features) + len(preprocessor.named_transformers_['cat'].categories_[0])])
print(processed_test_df.head())

print("\nData preprocessing complete. The processed data is ready for model training.")
