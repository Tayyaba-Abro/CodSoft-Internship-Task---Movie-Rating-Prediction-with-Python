# CodSoft Internship Task - Movie Rating Prediction with Python

## Introduction
Movie Rating Prediciton project involves building a model that predicts the rating of a movie based on features like genre, director, and actors. We use regression techniques to tackle this exciting problem. This enables us to explore data analysis, preprocessing, feature engineering, and machine learning modeling techniques

## Goal
The main goal of this project is to analyze historical movie data and develop a model that accurately estimates the rating given to a movie by users or critics. By doing so, we aim to provide insights into the factors that influence movie ratings and create a model that can estimate the ratings of movies accurately.

## Project Steps

### Importing Libraries

```python
# import necessary libraries required

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
```

### Reading Data
```python
# read the dataset into a dataframe
df = pd.read_csv("movies.csv", encoding='latin1')
# show first five records of dataframe
df.head()
```

### Data Wrangling
```python
# show the number of records and observations in the dataframe
df.shape

# check out the information on the dataframe
df.info()

# check out the missing values in each observation
df.isna().sum()

# drop records with missing value in any of the following columns: Name, Year, Duration, Votes, Rating
df.dropna(subset=['Name', 'Year', 'Duration', 'Votes', 'Rating'], inplace=True)

# check the missing values in each observation again
df.isna().sum()

# remove rows with duplicate movie records
df.drop_duplicates(subset=['Name', 'Year', 'Director'], keep='first', inplace=True)

# remove () from the Year column values and change the datatype to integer
df['Year'] = df['Year'].str.strip('()').astype(int)

# remove minutes from the Duration column values
df['Duration'] = df['Duration'].str.replace(r' min', '').astype(int)

# remove commas from Votes column and convert to integer

# show the number of records and observations after cleaning the dataframe
df.shape

# show the info on the cleaned dataframe
df.info()

# show the statistics of the dataframe
df.describe()
```

###Exploratory Data Analysis

#### i. Number of Movies each Year
```python
# group the data by Year and count the number of movies in each year
yearly_movie_counts = df['Year'].value_counts().sort_index()

# create a bar chart
plt.figure(figsize=(18, 9))
plt.bar(yearly_movie_counts.index, yearly_movie_counts.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Released Each Year')

# Show every second year on the x-axis and rotate x-labels for better readability
plt.xticks(yearly_movie_counts.index[::2], rotation=90)

plt.show()
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task---Movie-Rating-Prediction-with-Python/assets/47588244/e7f325a7-3bca-41c9-a586-210c512741f5)

#### ii. Creating Genre Dummy Columns
```python
# create dummy columns for each genre
dummies = df['Genre'].str.get_dummies(', ')
# creating a new dataframe which combines df and dummies
df_genre = pd.concat([df, dummies], axis=1)
```


