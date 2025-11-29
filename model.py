import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\pose\dataset_path\dataset_pathdataset.csv")

# Clean labels
df['label'] = df['label'].astype(str).str.strip().str.capitalize()

print("Unique labels found:", df['label'].unique())

# Map labels safely
y = df['label'].map({'Suspicious': 0, 'Normal': 1})

# Remove rows with missing target (NaN)
valid_index = y.dropna().index
df = df.loc[valid_index]
y = y.loc[valid_index]

# Prepare features
X = df.drop(['label', 'image_name'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=50,
    eval_metric='logloss',
    objective='binary:logistic',
    tree_method='hist',
    eta=0.1,
    max_depth=3,
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
model.save_model(r"C:\Users\DELL\Desktop\pose\trained_model.json")
