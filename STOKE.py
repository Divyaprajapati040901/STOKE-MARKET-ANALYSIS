import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("C:/Users/DIVYA/Desktop/PROJECT IIRS ISRO/Stoke/Adaniport.csv")

# Define features and target variable
X = data.drop(columns=['%Deliverble'])
y = data['%Deliverble']

# Define categorical and numerical features
categorical_features = ['Symbol','Day_name','Month_name']
numerical_features = ['Open',	'High',	'Low',	'Close',	'VWAP',	'Volume',	'Turnover',	'Trades',	'Deliverable_Volume']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor model
random_forest_model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor(random_state=42))])

# Train the Random Forest Regressor model
random_forest_model.fit(X_train, y_train)

# Predictions
y_pred = random_forest_model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Print the R2 score for Random Forest Regressor model
st.write(f'R2 Score for Random Forest Regressor: {r2}')

st.title('SALES RETAIL PRICE PREDICTION')

# Input fields for features

Symbol = st.selectbox('Select Symbol', data['Symbol'].unique())
Open = st.number_input('Open', value=0.0)
High = st.number_input('High', value=0.0)
Low = st.number_input('Low', value=0.0)
Close = st.number_input('Close', value=0.0)
VWAP = st.number_input('VWAP', value=0.0)
Volume = st.number_input('Volume', value=0.0)
Turnover = st.number_input('Turnover', value=0.0)
Trades = st.number_input('Trades', value=0.0)
Deliverable_Volume = st.number_input('Deliverable_Volume', value=0.0)
Day = st.number_input('Day', value=0)
Month = st.number_input('Month', value=0)
Year = st.number_input('Year', value=0)
Month_name = st.selectbox('Select Month_name', data['Month_name'].unique())
Day_name = st.selectbox('Select Day_name', data['Day_name'].unique())


# Prepare input features
input_features = pd.DataFrame({
    'Symbol': [Symbol],
    'Open': [Open],
    'High': [High],
    'Low': [Low],
    'Close': [Close],
    'VWAP': [VWAP],
    'Volume': [Volume],
    'Turnover': [Turnover],
    'Trades': [Trades],
    'Deliverable_Volume': [Deliverable_Volume],
    'Day': [Day],
    'Month': [Month],
    'Year': [Year],
    'Month_name': [Month_name],
    'Day_name': [Day_name],
})

# Predict button
if st.button('Predict'):
    # Predict using the trained model
    prediction = random_forest_model.predict(input_features)
    st.success(f'Predicted Sales Price: {prediction[0]}')
