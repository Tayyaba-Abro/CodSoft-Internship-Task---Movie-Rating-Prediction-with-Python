# CodSoft Internship Task - Movie Rating Prediction with Python

## Introduction
This project involves building a model that predicts the rating of a movie based on features like genre, director, and actors. We use regression techniques to tackle this exciting problem.

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

