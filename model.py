import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
df_red = pd.read_csv("winequality-red.csv", sep=";")
df_white = pd.read_csv("winequality-white.csv", sep=";")

# Add new attribute called wine_category to both dataframes
df_white['wine_category'] = 'white'
df_red['wine_category'] = 'red'

# Combine red and white wine data
df_wines = pd.concat([df_red, df_white])

# Convert quality into a categorical feature
df_wines['quality_label'] = df_wines['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_wines['quality_label'] = pd.Categorical(df_wines['quality_label'], categories=['low', 'medium', 'high'])

# Data encoding for the target variable
label_quality = LabelEncoder()
df_wines['quality_label_encoded'] = label_quality.fit_transform(df_wines['quality_label'])

# Define features (X) and target (y)
X = df_wines.drop(['quality', 'quality_label', 'quality_label_encoded', 'wine_category'], axis=1)  # Features
y = df_wines['quality_label_encoded']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model and LabelEncoder
with open('wine_quality_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_quality, le_file)
