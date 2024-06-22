import streamlit as st
import pandas as pd
import os

cwd = os.getcwd()

# Read CSV files with error_bad_lines=False
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

st.write("----------------------------------------------------------------------")

st.write("## Validation Data:")
st.dataframe(val_df.info())


st.write(f"There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in the training data.")
st.write(f"There are {test_df.shape[0]} rows and {test_df.shape[1]} columns in the testing data.")
st.write(f"There are {val_df.shape[0]} rows and {val_df.shape[1]} columns in the validation data.")


# checking for null values
print("Training Data:")
st.write(train_df.isnull().sum())
print("----------------------------------------------------------------------")
print("Testing Data:")
st.write(test_df.isnull().sum())
print("----------------------------------------------------------------------")
print("Validation Data:")
st.write(val_df.isnull().sum())


st.write('## Exploratory Data Analysis (EDA)')

st.write(train_df.head())
