
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

st.write(f"There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in the training data.")
st.write(f"There are {test_df.shape[0]} rows and {test_df.shape[1]} columns in the testing data.")
st.write(f"There are {val_df.shape[0]} rows and {val_df.shape[1]} columns in the validation data.")

# checking for null values
st.write("Training Data:")
st.write(train_df.isnull().sum())
st.write("----------------------------------------------------------------------")
st.write("Testing Data:")
st.write(test_df.isnull().sum())
st.write("----------------------------------------------------------------------")
st.write("Validation Data:")
st.write(val_df.isnull().sum())


st.write("I can do 2 things to deal with null values:")
st.write("Simple drop them as it is only 1 null value and might not affect our analysis.")
st.write("Fill them with the mean, median, or mode depending on the value of the column.")

st.write("## Exploratory Data Analysis (EDA)")
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

st.write("In the age coloum we have 4 irregular values.")
st.write("Male, Female, Non-binary, and other.")
st.write("Let's drop/fill these values first")

# removing the Male, Female, Non-binary, and işte mevcut veri kümesini 1000 satıra tamamlıyorum:

# Replace non-numeric values with NaN
train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')

# Handle NaN values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

train_df['Age'].unique()

st.title("Age Distribution")
fig = px.histogram(train_df, x='Age')
st.plotly_chart(fig)

st.title("Gender Distribution")
fig = px.histogram(train_df, x='Gender')
st.plotly_chart(fig)

st.write("We've the same issue here too. Mix of Gender with some number like in Age Coloum. Let's handle it.")

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
st.write(train_df['Gender'].unique())

st.title("Gender Distribution")
gender_fig = px.histogram(train_df, x='Gender')
st.plotly_chart(gender_fig)

st.write("### Platform Distribution")

train_df['Platform'].unique()
train_df['Platform'].value_counts()
st.write("Let's Impute the nan with the mode")
# filling with mode
train_df['Platform'].fillna(train_df['Platform'].mode()[0], inplace=True)


st.title("Platform Distribution")
Platform_fig = px.histogram(train_df, x='Platform')
st.plotly_chart(Platform_fig)

# st.write("### Daily Usage Time (minutes) Distribution")
st.title("Daily Usage Time (minutes) Distribution")
Daily_usage_fig = px.histogram(train_df, x='Daily_Usage_Time (minutes)')
st.plotly_chart(Daily_usage_fig)


# st.write("Post Per Day Distribution")
train_df['Posts_Per_Day'].unique()
# fill with mode
train_df['Posts_Per_Day'].fillna(train_df['Posts_Per_Day'].mode()[0], inplace=True)
st.title("Post Per Day Distribution")
Post_per_day_fig = px.histogram(train_df, x='Posts_Per_Day')
st.plotly_chart(Post_per_day_fig)


# st.write("Likes Per Day Distribution")
train_df['Likes_Received_Per_Day'].unique()
# filling wih mode
train_df['Likes_Received_Per_Day'].fillna(train_df['Likes_Received_Per_Day'].mode()[0], inplace=True)

st.title('Likes Per Day Distribution')
Like_per_day_fig = px.histogram(train_df, x = 'Likes_Received_Per_Day')
st.plotly_chart(Like_per_day_fig)

# st.write("Comments Per Day Distribution")
train_df['Comments_Received_Per_Day'].unique()
# filling with mode
train_df['Comments_Received_Per_Day'].fillna(train_df['Comments_Received_Per_Day'].mode()[0], inplace=True)
train_df['Comments_Received_Per_Day'].unique()

st.title('Posts Per Day Distribution')
Comment_received_fig = px.histogram(train_df, x = 'Comments_Received_Per_Day')
st.plotly_chart(Comment_received_fig)

# st.write("Messages Per Day Distribution")
st.title('Messages Per Day Distribution')
Message_per_day_fig = px.histogram(train_df, x = 'Messages_Sent_Per_Day')
st.plotly_chart(Message_per_day_fig)


# st.write("Emotion Distribution")
train_df['Dominant_Emotion'].unique()
# fill with mode
train_df['Dominant_Emotion'].fillna(train_df['Dominant_Emotion'].mode()[0], inplace=True)


plt = px.pie(train_df, names='Dominant_Emotion', title='Dominant Emotion Distribution')
plt.update_traces(textposition='inside', textinfo='percent+label')

# Display the Plotly chart in Streamlit
st.plotly_chart(plt)


st.write("Relationship Between Variables")
st.write("Gender and Platform")
# Group the data by gender and platform
grouped = train_df.groupby(['Gender', 'Platform'])

# Count the number of rows in each group
counts = grouped.size()

# Print the counts
st.write(counts)

st.title("Platform by Gender Usage")
Platform_by_gender_fig = px.histogram(train_df, x='Gender')
st.plotly_chart(Platform_by_gender_fig)

st.write("Age and Gender")
# grouping age with gender
grouped = train_df.groupby(['Age', 'Gender'])

# count the number of rows in each group
counts = grouped.size()

# print the counts
st.write(counts)

st.title("Age by Gender")
Age_by_gender_fig = px.histogram(train_df, x='Age')
st.plotly_chart(Age_by_gender_fig)

# st.write("Gender and Platform VS Daily Usage Time (minutes)")
st.title("Posts Per Day by Gender")
Posts_Per_Day_by_Gender_fig = px.histogram(train_df, x='Posts_Per_Day')
st.plotly_chart(Posts_Per_Day_by_Gender_fig)


st.write("Gender VS Emotions")
# checking the Gender againest the Dominant_Emotion
grouped = train_df.groupby(['Gender', 'Dominant_Emotion'])

# count the number of rows in each group
counts = grouped.size()

# print the counts
st.write(counts)
# ploting
st.title("Dominant Emotion by Gender")
Dominant_Emotion_by_Gender_fig = px.histogram(train_df, x='Gender')
st.plotly_chart(Dominant_Emotion_by_Gender_fig)


st.write("Platform VS Emotions")
# checking the Platformagainest the Dominant_Emotion
grouped = train_df.groupby(['Platform', 'Dominant_Emotion'])

# count the number of rows in each group
counts = grouped.size()

# print the counts
st.write(counts)
st.title("Dominant Emotion by Platform")
Dominant_Emotion_by_Platform_fig = px.histogram(train_df, x='Platform')
st.plotly_chart(Dominant_Emotion_by_Platform_fig)


# Create a contingency table
contingency_table = pd.crosstab(train_df['Platform'], train_df['Dominant_Emotion'])

fig = px.imshow(contingency_table, title='Platform vs Dominant Emotion Heatmap')
st.plotly_chart(fig)


# st.write("Time Spent VS Emotions")
# # Daily Usage time by Dominant_Emotion
st.title("Time Usage by Dominant Emotion")
Time_Usage_by_Dominant_Emotion_fig = px.histogram(train_df, x= 'Daily_Usage_Time (minutes)')
st.plotly_chart(Time_Usage_by_Dominant_Emotion_fig)


# st.write("Likes Received VS Emotions")
# st.write("This is the last realation I'll be looking in this data.")
st.title("Likes Received vs Dominant Emotion")
likes_vs_emotions_fig = px.histogram(train_df, x= 'Likes_Received_Per_Day')
st.plotly_chart(likes_vs_emotions_fig)


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

st.write("Logistic regression Classifier Report:")
st.write(classification_report(y_test, y_pred_lr))

# Random Forest Classifier
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

st.write("Random Forest Classifier Report:")
st.write(classification_report(y_test, y_pred_rf))

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

optionTime = st.sidebar.selectbox('opt', dailyUsageTimeDf['USAGE'])

genderDf = pd.DataFrame({
    'gender': [ 'Male', 'Female']
    })

gender = st.sidebar.selectbox('gender',
     genderDf['gender'])



# Example test data
new_data = pd.DataFrame({
    'Age': [optionAge],
    'Gender': [gender],
    'Platform': [optionPlatform],
    'Daily_Usage_Time (minutes)': [optionTime],
    'Posts_Per_Day': [3.0],
    'Likes_Received_Per_Day': [12.0],
    'Comments_Received_Per_Day': [10.0],
    'Messages_Sent_Per_Day': [12.0]
})

st.write(new_data)
# Assuming you have already trained your rf_pipeline
y_pred_new_data = rf_pipeline.predict(new_data)

# Print the predicted labels
st.write(y_pred_new_data)

cv_cross_validation = cross_validate(rf_pipeline, X_train, y_train, cv=5, scoring=['f1_weighted'], n_jobs=-1)
cv_cross_validation
cv_cross_validation['test_f1_weighted'].mean()
st.write(f"{cv_cross_validation['test_f1_weighted'].mean()} : cross validation result")

# Get cross-validated predictions
y_pred_cv = cross_val_predict(rf_pipeline, X_train, y_train, cv=5)

# Generate classification report
report = classification_report(y_train, y_pred_cv)
st.write(report)

st.write("Precision: It measures the accuracy of positive predictions. Formally, it is the number of true positives divided by the number of true positives plus the number of false positives. It’s a measure of a classifier’s exactness. A low precision can also indicate a large number of false positives.")
st.write("Recall: It measures the ability of a classifier to find all the positive samples. It is the number of true positives divided by the number of true positives plus the number of false negatives. It’s a measure of a classifier’s completeness. A low recall indicates many false negatives.")
st.write("F1-Score: It is the harmonic mean of precision and recall and provides a single score that balances both the concerns of precision and recall in one number. It’s a way to combine both metrics into one for easier comparison between classifiers.")


# Your existing pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Perform cross-validation
cv_results = cross_validate(rf_pipeline, X_train, y_train, cv=5)

# Print the cross-validation results
st.write(cv_results)


st.write("Insights:")
st.write("Age Distribution: We can see that the majority of the users are in all ages between 21 and 35. The higest count is in the age group of 27.")
st.write("Gender Distribution: We've 4 Values in the Gender Column. Male, Female, Non-Binary, Unknown.")
st.write("A. Male: 332,")
st.write("B. Female: 344,")
st.write("C. Non-Binary: 248,")
st.write("D .Unknown: 77")
st.write("Platform Distribution: We can see that the majority of the users use Instagram followed by Twitter and Facebook.")
st.write("A.Instagram 251")
st.write("B. Twitter 200")
st.write("C. Facebook 190")
st.write("D. LinkedIn 120")
st.write("E. Whatsapp 80")
st.write("F. Telegram 80")
st.write("G. Snapchat 80")
st.write("Daily_Usage_Time (minutes): The Daily Usage Time is from 40 to 200 minutes. Most count is in range of 60 to 90 minutes. While the Average time is 95 min.M")
st.write("Post per Day: The Minimum number of post is 1 and Maximum is 8, while the average is 3.32 posts. (Most Count is 2)")
st.write("Likes Received per Day: The Minimum number of like received is 5 and Highest is 110, while the average is 39.89.")
st.write("Comments Received per Day: The Minimum number of comment received is 2 and maximum is 40 while the average if 15.61.")
st.write("Messages Sent per Day: The Minimum number of messages sent is 8 and maximum is 50 while the average if 22.56.")
st.write("Emotions: We've total 6 emotions:")
st.write("A. Happiness: 20.1%")
st.write("B. Anger: 13%")
st.write("C. Neutral: 20%")
st.write("D. Anxiety: 17")
st.write("E. Boredom: 14%")
st.write("F. Sadness: 16%")
st.write("Gender VS Platform:")
st.write("A. Instagram is most usage app amoung female while facebook is the least one.")
st.write("B. Twitter is the most usage app amoung male followed by instagram. While Whatsapp is the least one.")
st.write("C. Facebook is the most usage app amoung non-binary.")
st.write("Post Per Day By Gender")
st.write("A. Instagram:")
st.write("a. Total Posts Per Day(Female): 885")
st.write("b. Total Posts Per Day(Female): 443")
st.write("B. Twitter/X:")
st.write("a. Total Posts Per Day(Female): 226")
st.write("b. Total Posts Per Day(Female): 351")
st.write("C. LinkedIn:")
st.write("a. Total Posts Per Day(Female): 60")
st.write("b. Total Posts Per Day(Female): 58")
st.write("Dominent Emotion by Gender:")
st.write("A. Female: The dominent emotion is Happniess and the count is 102. While Anger, Nuetral, Anxiety and Sadness are the other emotions with the almost same count range 56 - 48. The count of Boredom is 30 which is the least count amoung other.")
st.write("B. Male: The dominent emotion is Happniess and the count is 66. While Anger, Nuetral, Anxiety, Boredom and Sadness are the other emotions with the almost same count range 58 - 46.")
st.write("C. Non-Binary: The dominent emotion is Neutral and the count is 82. While Anxiety, Sadness and Boredom are the other emotions with the almost same count 46. Anger is the leaset count (10) among the other.")
st.write("Dominant_Emotion is strongly correlated with Platform.")
st.write("A. Happiness is the dominant emotion in the Instagram platform. While Anger is the least count.")
st.write("B. Anger is the dominant emotion in the Twitter platform and Whatsapp. While Happniess is the least count in both.")
st.write("C. Sadness is the also dominant emotion in the Twitter platform and Snapchat.")
st.write("D. Neutral is the dominant emotion in the Facebook platform and Telegram platform.")
st.write("E. Boredom is the dominant emotion in the LinkedIn platform and Facebook")
st.write("F. _Anixity- is the also dominant emotion in the Facebook and other platforms.")
st.write("Daily_Usage_Time (minutes) and Dominant_Emotion are strongly correlated.")
st.write("A. 200 minutes is related with Anxiety emotion.")
st.write("B. From 140 to 190 minutes is related with Happiness emotion.")
st.write("C. Anger is commonly seens in the range of 60 to 120 minutes.")
st.write("D. Other Emotions are also in the range of 40 to 130 minutes")
st.write("Likes Received Per Day is strongly correlated with Dominant_Emotion.")
st.write("A. Upon looking at the plot, we can see that the Happiness emotion is triggered by the Likes_Received_Per_Day Ranging from 60 to 109 Likes.")
st.write("B. While the Anxiety emotion is also triggered by the Likes_Received Ranging from 110 to 114, which could means more the likes received is causes the Anxiety.)")
