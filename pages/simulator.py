
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as snsb
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from pathlib import Path

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

st.header(":rainbow-background[Mental Health: Simulator]")


cwd = os.getcwd()

train_df = pd.read_csv(f'{cwd}/raw_data/train.csv',on_bad_lines='skip')
test_df = pd.read_csv(f'{cwd}/raw_data/test.csv', on_bad_lines='skip')
val_df = pd.read_csv(f'{cwd}/raw_data/val.csv', on_bad_lines='skip')


# Replace non-numeric values with NaN
train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')

# Handle NaN values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

train_df['Age'].unique()

#Function to replace numeric values with NaN
def clean_gender_column(gender_value):
    try:
        # Try converting the value to float, if it succeeds it's a numeric value
        float(gender_value)
        return np.nan
    except ValueError:
        # If conversion fails, it's a valid gender entry
        return gender_value

# Apply the function to the Gender column
train_df['Gender'] = train_df['Gender'].apply(clean_gender_column)

# Fill NaN values with a placeholder or drop them
# Here we fill NaN values with 'Unknown'
train_df['Gender'].fillna('Unknown', inplace=True)

    # Verify the unique values after cleaning

train_df['Platform'].unique()
train_df['Platform'].value_counts()
# filling with mode
train_df['Platform'].fillna(train_df['Platform'].mode()[0], inplace=True)


# st.write("Post Per Day Distribution")
train_df['Posts_Per_Day'].unique()
# fill with mode
train_df['Posts_Per_Day'].fillna(train_df['Posts_Per_Day'].mode()[0], inplace=True)


# st.write("Likes Per Day Distribution")
train_df['Likes_Received_Per_Day'].unique()
# filling wih mode
train_df['Likes_Received_Per_Day'].fillna(train_df['Likes_Received_Per_Day'].mode()[0], inplace=True)

train_df['Comments_Received_Per_Day'].unique()
# filling with mode
train_df['Comments_Received_Per_Day'].fillna(train_df['Comments_Received_Per_Day'].mode()[0], inplace=True)
train_df['Comments_Received_Per_Day'].unique()

# st.write("Emotion Distribution")
train_df['Dominant_Emotion'].unique()
# fill with mode
train_df['Dominant_Emotion'].fillna(train_df['Dominant_Emotion'].mode()[0], inplace=True)

# Group the data by gender and platform
grouped = train_df.groupby(['Gender', 'Platform'])

# Count the number of rows in each group
counts = grouped.size()

# grouping age with gender
grouped = train_df.groupby(['Age', 'Gender'])

# count the number of rows in each group
counts = grouped.size()


# checking the Gender againest the Dominant_Emotion
grouped = train_df.groupby(['Gender', 'Dominant_Emotion'])

# count the number of rows in each group
counts = grouped.size()


# checking the Platformagainest the Dominant_Emotion
grouped = train_df.groupby(['Platform', 'Dominant_Emotion'])

# count the number of rows in each group
counts = grouped.size()



# Create a contingency table
contingency_table = pd.crosstab(train_df['Platform'], train_df['Dominant_Emotion'])


# Checking for missing values
train_df.isnull().sum()

# Let's drop the missing values
train_df.dropna(inplace=True)

# Define features and target
X = train_df.drop(columns=['Dominant_Emotion', 'User_ID']) # Features
y = train_df['Dominant_Emotion'] # Target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps. Seeperating the numerical and categorical features
numeric_features = ['Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
categorical_features = ['Age', 'Gender', 'Platform']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Random Forest Classifier
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression())])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)


# Random Forest Classifier
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

ageDF = pd.DataFrame({
    'AGE': [ 21, 22, 23, 24, 25, 26,27,28,29,30]
    })

optionAge = st.sidebar.selectbox(
    'Age?',
     ageDF['AGE'])


platformDF = pd.DataFrame({
    'AGE': ['Twitter', 'Facebook', 'LinkedIn', 'Snapchat', 'Whatsapp', 'Telegram', 'Instagram']
    })

optionPlatform = st.sidebar.selectbox(
    'Platform',
     platformDF['AGE'])



dailyUsageTimeDf = pd.DataFrame({
    'USAGE': [ 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    })

optionTime = st.sidebar.selectbox('Daily Usage time (mins)', dailyUsageTimeDf['USAGE'])

genderDf = pd.DataFrame({
    'gender': [ 'Male', 'Female', 'Non-binary', 'Unknown']
    })

gender = st.sidebar.selectbox('gender',
     genderDf['gender'])


postsDF = pd.DataFrame({
    'POSTS_PER_DAY': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'LIKES_PER_DAY': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
    'COMMENTS_PER_DAY': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
    'MESSAGES_SENT_PER_DAY': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
})

postsPerDay = st.sidebar.selectbox('Posts per day', postsDF['POSTS_PER_DAY'])
likesPerDay = st.sidebar.selectbox('Likes per day', postsDF['LIKES_PER_DAY'])
commentsPerDay = st.sidebar.selectbox('Comments per day', postsDF['COMMENTS_PER_DAY'])
messagesSentPerDay = st.sidebar.selectbox('Messages sent per day', postsDF['MESSAGES_SENT_PER_DAY'])

# Example test data
new_data = pd.DataFrame({
    'Age': [optionAge],
    'Gender': [gender],
    'Platform': [optionPlatform],
    'Daily_Usage_Time (minutes)': [optionTime],
    'Posts_Per_Day': [postsPerDay],
    'Likes_Received_Per_Day': [likesPerDay],
    'Comments_Received_Per_Day': [commentsPerDay],
    'Messages_Sent_Per_Day': [messagesSentPerDay]
})

st.write(new_data)
# Assuming you have already trained your rf_pipeline
y_pred_new_data = rf_pipeline.predict(new_data)
image_width = 150  # Set the desired width to make the image smaller
image_path_sadness = 'https://raw.githubusercontent.com/Diane077/mental_health/master/images/sad.png'
image_path_anger = 'https://raw.githubusercontent.com/Diane077/mental_health/master/images/anger.png'
image_path_anxiety = 'https://raw.githubusercontent.com/Diane077/mental_health/master/images/anxiety.png'
image_path_bored = 'https://raw.githubusercontent.com/Diane077/mental_health/master/images/bored.png'
image_path_happy = 'https://raw.githubusercontent.com/Diane077/mental_health/master/images/happy.png'
image_path_neutral = 'https://raw.githubusercontent.com/Diane077/mental_health/master/images/neutral.png'

# Print the predicted labels
st.write(y_pred_new_data[0])
if y_pred_new_data[0] == "Sadness":
    st.markdown(f"<div style='text-align: center'><img src='{image_path_sadness}' width='{image_width}'></div>", unsafe_allow_html=True)

if y_pred_new_data[0] == "Anger":
    st.markdown(f"<div style='text-align: center'><img src='{image_path_anger}' width='{image_width}'></div>", unsafe_allow_html=True)

if y_pred_new_data[0] == "Anxiety":
    st.markdown(f"<div style='text-align: center'><img src='{image_path_anxiety}' width='{image_width}'></div>", unsafe_allow_html=True)

if y_pred_new_data[0] == "Boredom":
    st.markdown(f"<div style='text-align: center'><img src='{image_path_bored}' width='{image_width}'></div>", unsafe_allow_html=True)

if y_pred_new_data[0] == "Happiness":
    st.markdown(f"<div style='text-align: center'><img src='{image_path_happy}' width='{image_width}'></div>", unsafe_allow_html=True)

if y_pred_new_data[0] == "Neutral":
    st.markdown(f"<div style='text-align: center'><img src='{image_path_neutral}' width='{image_width}'></div>", unsafe_allow_html=True)
