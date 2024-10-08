# -*- coding: utf-8 -*-
"""data_cleaning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DxM9wMk_CYfkl9EpTSzyhUUMfeJHlNrk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# prompt: upload file from drive

from google.colab import files
uploaded = files.upload()

# prompt: load uploaded csv fie using panda
df = pd.read_csv('Car Fuel and Emissions 2000-2013.csv')
df.head()

df.info()

df.columns.str.strip() ### to remove any wide spaces from column names

### Now we will create heat map to check correlation between different numeric colums
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

"""##Based on results we will only choose those columns for analysis which are having significant correaltion with Fuel_cost_6000 miles, we will drop all other columns. Also from categorical column we will only be selecting fuel type."""

df_analysis= df[['year','euro_standard','engine_capacity','co2','combined_metric','combined_imperial','fuel_type','fuel_cost_6000_miles']]
df_analysis.head()

"""##Now we will check these columns for null values and we will impute null values or drop rows with null values"""

df_analysis.isnull().sum()

"""Above result shows we have significant null values from the "Fuel_cost_6000_miles" column and it is that column which value we intend to predict using Machine learning, so we cant assign value from our side other wise it will jeopardize the results. removing these values will result in significant data loss but we dont have choice right now."""

df_analysis.dropna(subset=['fuel_cost_6000_miles'], inplace=True)

df_analysis.isnull().sum()

df_analysis.info()

"""**Now we have total 7 columns with non null values of 12200... we will explore each column further to determine if there is any outlier**"""

df_analysis.year.unique()

df_analysis.year.value_counts() ### we have considerable number of rows from each year so there is no outlier

df_analysis.euro_standard.unique()

df_analysis.euro_standard.value_counts() ## Each value has significant number of rows so no outliers in this case as well.

df_analysis.engine_capacity.unique()

sns.boxplot(df_analysis.engine_capacity)

"""we can see max values are going out of the range and can be considered as outlier"""

df_analysis.engine_capacity.max()

df_analysis.loc[df_analysis.engine_capacity ==7990.0]

df_analysis=df_analysis[df_analysis.engine_capacity != df_analysis.engine_capacity.max()]

df_analysis.co2.unique()

sns.boxplot(df_analysis.co2)  ## result shows values on maximum side are in significant number and cant be considered as outlier

df_analysis.combined_metric.unique()

sns.boxplot(df_analysis.combined_metric) ## result shows values on maximum side are in significant number and cant be considered as outlier

df_analysis.combined_imperial.unique()

sns.boxplot(df_analysis.combined_imperial) ## max value in this case can be considered as outlier

df_analysis.combined_imperial.max()

df_analysis.loc[df_analysis['combined_imperial']==83.1] ## there are several values at maximum so we are not removing them

df_analysis.fuel_type.unique()

df_analysis.fuel_type.value_counts() ## we have low number of rows with CNG and Petrol hybrid, we explore these columns further

df_analysis.loc[df_analysis['fuel_type']=='CNG']

"""we discovered another thing, all values of fuel_Cost_6000 miles corressponding to our fuel type CNG is 0, which is not possible , hence these values are incorrectly recorded / stored and unfortunately we will have to drop fuel_type 'CNG'"""

df_analysis=df_analysis[df_analysis.fuel_type != 'CNG']

df_analysis[df_analysis.fuel_type == 'Petrol Hybrid']

df_analysis=df_analysis[df_analysis.fuel_type != 'Petrol Hybrid']  ### As above result shows we have only 1 non zero value of fuel cost corresponding to Petrol Hybrid type

df_analysis.info()

df_analysis.fuel_type.value_counts()

"""##Now encoding fuel_type column"""

df_encoded = pd.get_dummies(df_analysis, columns=['fuel_type'])
df_encoded.head()

"""## Now we have removed null values, outliers and encoded Fuel_type column our dataset is ready for model training."""

df_encoded.to_csv('df_encoded.csv', encoding = 'utf-8-sig') ## To save our data in Csv format
files.download('df_encoded.csv')