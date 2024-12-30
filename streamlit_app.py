import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# Page configuration
st.set_page_config(
    page_title='Binary Classification App',
    page_icon='🌸',
    layout='wide',
    initial_sidebar_state='expanded')

# Title
st.title('Binary Classification: Setosa vs Not Setosa')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/Vonmonyroth/Book-Anlyst/refs/heads/main/iris.csv')

# Convert species to binary classification: Setosa or Not Setosa
df['Species'] = df['Species'].apply(lambda x: 1 if x == 'setosa' else 0)

# Input widgets
st.sidebar.subheader('Input Features')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)

# Model selection
model_choice = st.sidebar.selectbox('Select Model', ['Logistic Regression (Scikit-learn)', 'Logistic Regression (PyTorch)'])

# Prepare data
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression using Scikit-learn
if model_choice == 'Logistic Regression (Scikit-learn)':
    model = SklearnLogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_class = 'Setosa' if prediction[0] == 1 else 'Not Setosa'

# Logistic Regression using PyTorch
elif model_choice == 'Logistic Regression (PyTorch)':
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, output_size)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # Model setup
    input_size = X_train.shape[1]
    output_size = 1
    model = LogisticRegressionModel(input_size, output_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Make prediction
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
        output = model(input_tensor)
        prediction = 1 if output.item() >= 0.5 else 0
        predicted_class = 'Setosa' if prediction == 1 else 'Not Setosa'

st.subheader('Brief EDA')
st.write('The data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)

st.write('Note! 1 is setosa and 0 is not setosa')

# Display input features
st.subheader('Input Features')
st.write(pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))

# Display prediction
st.subheader('Prediction Output')
st.metric('Predicted Class', predicted_class)
