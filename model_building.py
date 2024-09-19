# -*- coding: utf-8 -*-
"""Model_building.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EHBjc7JWBeEhmRTd0BpkfpLIXO9o4dl_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

import io
df = pd.read_csv(io.BytesIO(uploaded['df_encoded.csv']))
df.head()



"""**Now we will split data in X-train, y_train, X_test, y_test and will apply linear regression to predict fuel_price_6000_miles , also we will evaluate our model in the end **"""

features = ['year','euro_standard','engine_capacity','co2','combined_metric','combined_imperial','fuel_type_Diesel','fuel_type_LPG','fuel_type_Petrol']
X = df[features]
y = df['fuel_cost_6000_miles']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae= mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error:{mae}')
print(f'R-squared: {r2}')

"""##Accuracy is good, and error is minimum so over choice of machine learning model is correct. Now we wil plot grpahs to visualize actual and predicted values"""

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
ax1.scatter(X_test['year'], y_test, label='Actual', alpha=0.7)
ax1.scatter(X_test['year'], y_pred, label='Predicted', alpha=0.7)
ax1.set_xlabel('Year')
ax1.set_ylabel('Fuel Cost (6000 miles)')
ax1.set_title('Year vs Fuel Cost')
ax1.legend()

ax2.scatter(X_test['engine_capacity'], y_test, label='Actual', alpha=0.7)
ax2.scatter(X_test['engine_capacity'], y_pred, label='Predicted', alpha=0.7)
ax2.set_xlabel('Engine Capacity')
ax2.set_ylabel('Fuel Cost (6000 miles)')
ax2.set_title('Engine Capacity vs Fuel Cost')
ax2.legend()

plt.tight_layout()
ax3.scatter(X_test['co2'], y_test, label='Actual', alpha=0.7)
ax3.scatter(X_test['co2'], y_pred, label='Predicted', alpha=0.7)
ax3.set_xlabel('CO2')
ax3.set_ylabel('Fuel Cost (6000 miles)')
ax3.set_title('CO2 vs Fuel Cost')
ax3.legend()


plt.tight_layout()
ax4.scatter(X_test['euro_standard'], y_test, label='Actual', alpha=0.7)
ax4.scatter(X_test['euro_standard'], y_pred, label='Predicted', alpha=0.7)
ax4.set_xlabel('euro_standard')
ax4.set_ylabel('Fuel Cost (6000 miles)')
ax4.set_title('euro_standard vs Fuel Cost')
ax4.legend()

"""## Now we will save the model"""

import joblib
joblib.dump(model, 'fuel_cost_6000_miles.joblib')

from google.colab import drive
drive.mount('/content/drive')

joblib.dump(model, '/content/drive/MyDrive/ML Models/fuel_cost_6000_miles.sav') #sav