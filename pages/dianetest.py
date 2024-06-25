
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
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

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



"""**Data loading**"""
""" * We've 3 files:
    - test.csv
    - train.csv
    - val.csv """



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

st.write("Gender Distribution")
train_df['Gender'].unique()
st.write("We've the same issue here too. Mix of Gender with some number like in Age Coloum. Let's handle it.")
# Function to replace numeric values with NaN
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
print(train_df['Gender'].unique())

train_df['Gender'].unique()
train_df['Gender'].value_counts()
plt = px.histogram(train_df, x='Gender', title='Gender Distribution')
plt.show()

st.write("Platform Distribution")
train_df['Platform'].unique()
train_df['Platform'].value_counts()
st.write("Let's Impute the nan with the mode")
# filling with mode
train_df['Platform'].fillna(train_df['Platform'].mode()[0], inplace=True)
train_df['Platform'].unique()
train_df['Platform'].value_counts()
plt = px.histogram(train_df, x='Platform', title='Platform Distribution')
plt.show()

st.write("Daily Usage Time (minutes) Distribution")
plt = px.histogram(train_df, x='Daily_Usage_Time (minutes)', title='Daily Usage Time Distribution')
plt.show()

st.write("Post Per Day Distribution")
train_df['Posts_Per_Day'].unique()
# fill with mode
train_df['Posts_Per_Day'].fillna(train_df['Posts_Per_Day'].mode()[0], inplace=True)
plt = px.histogram(train_df, x='Posts_Per_Day', title='Posts Per Day Distribution')
plt.show()

st.write("Likes Per Day Distribution")
train_df['Likes_Received_Per_Day'].unique()
# filling wih mode
train_df['Likes_Received_Per_Day'].fillna(train_df['Likes_Received_Per_Day'].mode()[0], inplace=True)
plt = px.histogram(train_df, x='Likes_Received_Per_Day', title='Posts Per Day Distribution')
plt.show()

st.write("Comments Per Day Distribution")
train_df['Comments_Received_Per_Day'].unique()
# filling with mode
train_df['Comments_Received_Per_Day'].fillna(train_df['Comments_Received_Per_Day'].mode()[0], inplace=True)
train_df['Comments_Received_Per_Day'].unique()
plt = px.histogram(train_df, x='Comments_Received_Per_Day', title='Posts Per Day Distribution')
plt.show()

st.write("Messages Per Day Distribution")
plt = px.histogram(train_df, x='Messages_Sent_Per_Day', title='Posts Per Day Distribution')
plt.show()

st.write("Emotion Distribution")
train_df['Dominant_Emotion'].unique()
# fill with mode
train_df['Dominant_Emotion'].fillna(train_df['Dominant_Emotion'].mode()[0], inplace=True)
plt = px.pie(train_df, names='Dominant_Emotion', title='Dominant Emotion Distribution')
# adding the values to the pie section
plt.update_traces(textposition='inside', textinfo='percent+label')
plt.show()

st.write("Relationship Between Variables")
st.write("Gender and Platform")
# Group the data by gender and platform
grouped = train_df.groupby(['Gender', 'Platform'])

# Count the number of rows in each group
counts = grouped.size()

# Print the counts
print(counts)
plt = px.histogram(train_df, x='Gender', color='Platform', title='Platform by Gender Usage')
plt.show()

st.write("Age and Gender")
# grouping age with gender
grouped = train_df.groupby(['Age', 'Gender'])

# count the number of rows in each group
counts = grouped.size()

# print the counts
print(counts)
plt = px.histogram(train_df, x='Age', color='Gender', title='Age by Gender')
plt.show()

st.write("Gender and Platform VS Daily Usage Time (minutes)")
plt = px.histogram(train_df, x='Posts_Per_Day', y='Platform' ,color='Gender', title='Posts Per Day by Gender')
plt.show()

st.write("Gender VS Emotions")
# checking the Gender againest the Dominant_Emotion
grouped = train_df.groupby(['Gender', 'Dominant_Emotion'])

# count the number of rows in each group
counts = grouped.size()

# print the counts
print(counts)
# ploting
plt = px.histogram(train_df, x='Gender', color='Dominant_Emotion', title='Dominant Emotion by Gender')
plt.show()
st.write("Platform VS Emotions")
# checking the Platformagainest the Dominant_Emotion
grouped = train_df.groupby(['Platform', 'Dominant_Emotion'])

# count the number of rows in each group
counts = grouped.size()

# print the counts
print(counts)
Plt = px.histogram(train_df, x='Platform', color='Dominant_Emotion', title='Dominant Emotion by Platform')
Plt.show()
# Create a contingency table
contingency_table = pd.crosstab(train_df['Platform'], train_df['Dominant_Emotion'])

# Plot the heatmap
fig = px.imshow(contingency_table, title='Platform vs Dominant Emotion Heatmap')
fig.show()

st.write("Time Spent VS Emotions")
# Daily Usage time by Dominant_Emotion
plt = px.histogram(train_df, x='Daily_Usage_Time (minutes)', color='Dominant_Emotion', title='Time Usage by Dominant Emotion')
plt.show()

st.write("Likes Received VS Emotions")
st.write("This is the last realation I'll be looking in this data.")
plt = px.histogram(train_df, x='Likes_Received_Per_Day', color='Dominant_Emotion', title='Likes Received vs Dominant Emotion')
plt.show()


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

st.write("Random Forest Classifier")
st.write("Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification). It improves predictive performance and reduces overfitting compared to a single decision tree by averaging the results of many trees.")

# Random Forest Classifier
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression())])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

print("Logistic regression Classifier Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest Classifier
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("Random Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))

cv_cross_validation = cross_validate(rf_pipeline, X_train, y_train, cv=5, scoring=['f1_weighted'], n_jobs=-1)
cv_cross_validation
cv_cross_validation['test_f1_weighted'].mean()


# Get cross-validated predictions
y_pred_cv = cross_val_predict(rf_pipeline, X_train, y_train, cv=5)

# Generate classification report
report = classification_report(y_train, y_pred_cv)
print(report)

st.write("Precision: It measures the accuracy of positive predictions. Formally, it is the number of true positives divided by the number of true positives plus the number of false positives. It’s a measure of a classifier’s exactness. A low precision can also indicate a large number of false positives.")
st.write("Recall: It measures the ability of a classifier to find all the positive samples. It is the number of true positives divided by the number of true positives plus the number of false negatives. It’s a measure of a classifier’s completeness. A low recall indicates many false negatives.")
st.write("F1-Score: It is the harmonic mean of precision and recall and provides a single score that balances both the concerns of precision and recall in one number. It’s a way to combine both metrics into one for easier comparison between classifiers.")


# Your existing pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Perform cross-validation
cv_results = cross_validate(rf_pipeline, X_train, y_train, cv=5)

# Print the cross-validation results
print(cv_results)

st.write("XGBoost Classifier")
st.write("XGBoost (Extreme Gradient Boosting) is a scalable and efficient implementation of gradient boosting algorithms. It builds an ensemble of trees in a sequential manner, where each tree tries to correct the errors of the previous trees. XGBoost is known for its high performance and speed, making it popular for structured or tabular data tasks in machine learning competitions and real-world applications.")
from sklearn.preprocessing import LabelEncoder
# Encode target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42))])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

# Decode the predicted labels back to original string labels for the report
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

print("XGBoost Classifier Report:")
print(classification_report(y_test_decoded, y_pred_xgb_decoded))

st.write("Neural Network")
# Define features and target
X = train_df.drop(columns=['Dominant_Emotion', 'User_ID'])
y = train_df['Dominant_Emotion']

# Encode target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode the target labels for neural network
y_onehot = to_categorical(y_encoded)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Define preprocessing steps for features
numeric_features = ['Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
categorical_features = ['Age', 'Gender', 'Platform']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Preprocess the features
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

st.write("Build the neural network")
# Build the neural network
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_onehot.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Decode the predicted and true labels back to original string labels for the report
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
y_test_labels = label_encoder.inverse_transform(y_test_classes)

# Print classification report
print("Neural Network Classifier Report:")
print(classification_report(y_test_labels, y_****dpred_labels))

st.write("Insights:")
st.write("Age Distribution: We can see that the majority of the users are in all ages between 21 and 35. The higest count is in the age group of 27.
Gender Distribution: We've 4 Values in the Gender Column. Male, Female, Non-Binary, Unknown.
A. Male: 332,
B. Female: 344,
C. Non-Binary: 248,
D .Unknown: 77
Platform Distribution: We can see that the majority of the users use Instagram followed by Twitter and Facebook.
A.Instagram 251
B. Twitter 200
C. Facebook 190
D. LinkedIn 120
E. Whatsapp 80
F. Telegram 80
G. Snapchat 80
Daily_Usage_Time (minutes): The Daily Usage Time is from 40 to 200 minutes. Most count is in range of 60 to 90 minutes. While the Average time is 95 min.M
Post per Day: The Minimum number of post is 1 and Maximum is 8, while the average is 3.32 posts. (Most Count is 2)
Likes Received per Day: The Minimum number of like received is 5 and Highest is 110, while the average is 39.89.
Comments Received per Day: The Minimum number of comment received is 2 and maximum is 40 while the average if 15.61.
Messages Sent per Day: The Minimum number of messages sent is 8 and maximum is 50 while the average if 22.56.
Emotions: We've total 6 emotions:
A. Happiness: 20.1%
B. Anger: 13%
C. Neutral: 20%
D. Anxiety: 17
E. Boredom: 14%
F. Sadness: 16%
Gender VS Platform:
A. Instagram is most usage app amoung female while facebook is the least one.
B. Twitter is the most usage app amoung male followed by instagram. While Whatsapp is the least one.
C. Facebook is the most usage app amoung non-binary.
Post Per Day By Gender
A. Instagram:
a. Total Posts Per Day(Female): 885
b. Total Posts Per Day(Female): 443
B. Twitter/X:
a. Total Posts Per Day(Female): 226
b. Total Posts Per Day(Female): 351
C. LinkedIn:
a. Total Posts Per Day(Female): 60
b. Total Posts Per Day(Female): 58
Dominent Emotion by Gender:
A. Female: The dominent emotion is Happniess and the count is 102. While Anger, Nuetral, Anxiety and Sadness are the other emotions with the almost same count range 56 - 48. The count of Boredom is 30 which is the least count amoung other.
B. Male: The dominent emotion is Happniess and the count is 66. While Anger, Nuetral, Anxiety, Boredom and Sadness are the other emotions with the almost same count range 58 - 46.
C. Non-Binary: The dominent emotion is Neutral and the count is 82. While Anxiety, Sadness and Boredom are the other emotions with the almost same count 46. Anger is the leaset count (10) among the other.
Dominant_Emotion is strongly correlated with Platform.
A. Happiness is the dominant emotion in the Instagram platform. While Anger is the least count.
B. Anger is the dominant emotion in the Twitter platform and Whatsapp. While Happniess is the least count in both.
C. Sadness is the also dominant emotion in the Twitter platform and Snapchat.
D. Neutral is the dominant emotion in the Facebook platform and Telegram platform.
E. Boredom is the dominant emotion in the LinkedIn platform and Facebook
F. _Anixity- is the also dominant emotion in the Facebook and other platforms.
Daily_Usage_Time (minutes) and Dominant_Emotion are strongly correlated.
A. 200 minutes is related with Anxiety emotion.
B. From 140 to 190 minutes is related with Happiness emotion.
C. Anger is commonly seens in the range of 60 to 120 minutes.
D. Other Emotions are also in the range of 40 to 130 minutes
Likes Received Per Day is strongly correlated with Dominant_Emotion.
A. Upon looking at the plot, we can see that the Happiness emotion is triggered by the Likes_Received_Per_Day Ranging from 60 to 109 Likes.
B. While the Anxiety emotion is also triggered by the Likes_Received Ranging from 110 to 114, which could means more the likes received is causes the Anxiety.)
