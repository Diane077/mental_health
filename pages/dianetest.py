
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

st.header(":rainbow-background[Project Title: Mental Health]")
""" **About Dataset** """
st.write("This unique dataset was meticulously researched and prepared by AI Inventor Emirhan BULUT. It captures valuable information on social media usage and the dominant emotional state of users based on their activities. The dataset is ideal for exploring the relationship between social media usage patterns and emotional well-being. Features:")



st.write("**Files:**")
st.write("* train.csv: Data for training models")
st.write("* test.csv: Data for testing models.")
st.write("* val.csv: Data for validation purposes.")

st.write("**Dataset Overview:**")
st.write(":globe_with_meridians: Source: **Dataset is taken from** :blue[Kaggle]")
st.write("**Columns:**")
st.write("* User_ID: Unique identifier for the user.")
st.write("* Age: Age of the user.")
st.write("* Gender: Gender of the user (Female, Male, Non-binary).")
st.write("* Platform: Social media platform used (e.g., Instagram, Twitter, Facebook, LinkedIn, Snapchat, Whatsapp, Telegram).")
st.write("* Daily_Usage_Time (minutes): Daily time spent on the platform in minutes.")
st.write("* Posts_Per_Day: Number of posts made per day.")
st.write("* Likes_Received_Per_Day: Number of likes received per day.")
st.write("* Comments_Received_Per_Day: Number of comments received per day.")
st.write("* Messages_Sent_Per_Day: Number of messages sent per day.")
st.write("* Dominant_Emotion: User's dominant emotional state during the day (e.g., Happiness, Sadness, Anger, Anxiety, Boredom, Neutral).")

"""**Introduction:**"""
"""Social media is a central point of our life. It has changed the way we live, work, and interact with others. As a result, it has become an essential part of our daily lives. The dataset here aims to provide valuable insights into social media usage and the dominant emotional state of users"""
"""The goal of this project is to analyze the relation between social media trends and emotional well-being, and train the models."""

"""**Importing Libraries**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsb
import plotly.express as px

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

"""**Data loading**"""
""" * We've 3 files:
    - test.csv
    - train.csv
    - val.csv """

import os

cwd = os.getcwd()

train_df = pd.read_csv(f'{cwd}/raw_data/train.csv',on_bad_lines='skip')
test_df = pd.read_csv(f'{cwd}/raw_data/test.csv', on_bad_lines='skip')
val_df = pd.read_csv(f'{cwd}/raw_data/val.csv', on_bad_lines='skip')

# Display data
st.write("Training Data:")
st.write(train_df.head())
st.write("----------------------------------------------------------------------")
st.write("Testing Data:")
st.write(test_df.head())
st.write("----------------------------------------------------------------------")
st.write("Validation Data:")
st.write(val_df.head())

# training data info
print("Traing Data Info:")
train_df.info()

# testing data info
print("Testing Data Info:")
test_df.info()


# validation data info
print("Validation Data Info:")
val_df.info()

st.write("Shape of the training, testing, and validation data.")

print(f"There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in the training data.")
print(f"There are {test_df.shape[0]} rows and {test_df.shape[1]} columns in the testing data.")
print(f"There are {val_df.shape[0]} rows and {val_df.shape[1]} columns in the validation data.")

# checking for null values
print("Training Data:")
display(train_df.isnull().sum())
print("----------------------------------------------------------------------")
print("Testing Data:")
display(test_df.isnull().sum())
print("----------------------------------------------------------------------")
print("Validation Data:")
display(val_df.isnull().sum())


st.write("I can do 2 things to deal with null values:")
st.write("Simple drop them as it is only 1 null value and might not affect our analysis.")
st.write("Fill them with the mean, median, or mode depending on the value of the column.")

st.write("Exploratory Data Analysis (EDA)")
st.write("Let's start EDA with the training data.")

train_df.head()

st.write("Column in our data:")

# list of coloums in the training data
train_df.columns

st.write("Describe the training data")
train_df.describe()

st.write("Age Distribution")
train_df['Age'].isnull().sum()
train_df['Age'].unique()

st.write("In the age coloum we have 4 irrigular values.")
st.write("Male, Female, Non-binary, and other.")
st.write("Let's drop/fill these values first")

# removing the Male, Female, Non-binary, and işte mevcut veri kümesini 1000 satıra tamamlıyorum:

# Replace non-numeric values with NaN
train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')

# Handle NaN values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

train_df['Age'].unique()
plt = px.histogram(train_df, x='Age', title='Age Distribution')
plt.show()
