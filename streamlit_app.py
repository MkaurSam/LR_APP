import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample Data (Replace with your actual dataset)
data = {
    'Payment_Delay': [1.5, 2.3, 3.1, 4.7, 1.8, 3.9],
    'Payment_Term': [2.1, 3.4, 1.7, 3.6, 4.1, 2.5],
    'Past_Due': [0, 1, 0, 1, 0, 1]  # 0 = Not Past Due, 1 = Past Due
}
df = pd.DataFrame(data)

# Splitting data into features and target
X = df[['Payment_Delay', 'Payment_Term']]  # Replace with your feature columns
y = df['Past_Due']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title('Customer Past Due Prediction')

# Input features from user
st.subheader('Enter Customer Features:')
feature1 = st.number_input('Payment_Delay', min_value=0.0, max_value=10.0, step=0.1)
feature2 = st.number_input('Payment_Term', min_value=0.0, max_value=10.0, step=0.1)

# Predict Button
if st.button('Predict'):
    # Make prediction based on user input
    user_input = pd.DataFrame([[feature1, feature2]], columns=['Payment_Delay', 'Payment_Term'])
    prediction = model.predict(user_input)
    
    # Show result
    if prediction[0] == 1:
        st.success('The customer is predicted to be Past Due')
    else:
        st.success('The customer is predicted to be Not Past Due')

# Model accuracy
st.subheader('Model Accuracy:')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')
