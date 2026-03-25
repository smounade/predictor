from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==============================
# 1. Load & preprocess dataset
# ==============================

data = pd.read_csv('properties.csv')

data.drop_duplicates(inplace=True)

# Convert relevant columns to Int64
to_int_cols = [
    'Postal code', 'Price', 'Number of rooms', 'Living Area',
    'Fully equipped kitchen', 'Furnished', 'Open fire', 'Terrace', 'Terrace Area',
    'Garden', 'Garden Area', 'Surface of the land', 'Number of facades', 'Swimming pool'
]

for col in to_int_cols:
    if col in data.columns:
        data[col] = data[col].astype('Int64')

# Drop rows with missing critical values
for col in ['Price', 'Number of rooms', 'Living Area', 'Surface of the land']:
    data.dropna(subset=[col], inplace=True)

# Fill NaN values in categorical/binary columns
fillna_defaults = {
    'Fully equipped kitchen': 1,
    'Furnished': 0,
    'Open fire': 0,
    'Terrace': 0,
    'Terrace Area': lambda df: df['Terrace Area'].median(),
    'Garden': 0,
    'Garden Area': lambda df: df['Garden Area'].median(),
    'Number of facades': lambda df: df['Number of facades'].median(),
    'Swimming pool': 0,
    'State of the building': 'Normal'  # will be dropped anyway
}

for col, val in fillna_defaults.items():
    if col in data.columns:
        if callable(val):
            data[col] = data[col].fillna(val(data))
        else:
            data[col] = data[col].fillna(val)


# Drop unnecessary columns 
drop_cols = ['Locality', 'Type of property', 'Subtype of property', 'Terrace', 'Garden','State of the building', 'url', 'title']
for col in drop_cols:
    if col in data.columns:
        data.drop(columns=[col], inplace=True)

# ==============================
# 2. Prepare data and rename columns with underscores
# ==============================

X = data.drop(columns=['Price', 'Postal code'])  # Postal code excluded from features
y = data['Price']

# Rename columns: replace spaces with underscores for consistency
X.columns = [col.replace(' ', '_') for col in X.columns]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

feature_cols = list(X_train.columns)  # All feature columns with underscores

# ==============================
# 3. Compute mean/std and scale features
# ==============================

means = X_train.mean()
stds = X_train.std()

X_train_std = (X_train - means) / stds
X_test_std = (X_test - means) / stds

# ==============================
# 4. Train Linear Regression model on scaled data
# ==============================

regressor = LinearRegression()
regressor.fit(X_train_std, y_train)

print(f"Train R^2 score: {regressor.score(X_train_std, y_train):.3f}")
print(f"Test R^2 score: {regressor.score(X_test_std, y_test):.3f}")

# ==============================
# 5. Define FastAPI app and Pydantic model
# ==============================

app = FastAPI()

class PropertyFeatures(BaseModel):
    Number_of_rooms: int
    Living_Area: int
    Fully_equipped_kitchen: int
    Furnished: int
    Open_fire: int
    Terrace_Area: int
    Garden_Area: int
    Surface_of_the_land: int
    Number_of_facades: int
    Swimming_pool: int

# ==============================
# 6. Prediction endpoint
# ==============================

@app.post("/predict")
def predict(data: PropertyFeatures):
    input_df = pd.DataFrame([data.model_dump()])
    
    # Ensure columns order matches training features
    input_df = input_df[feature_cols]
    
    # Standardize input
    input_std = (input_df - means) / stds
    
    # Predict
    prediction = regressor.predict(input_std)[0]
    
    return {"predicted_price": round(float(prediction), 2)}

# ==============================
# 7. Run Uvicorn server if standalone
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
