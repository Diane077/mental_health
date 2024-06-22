
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

st.header("Project Title: Mental Health")
st.write("About Dataset")
This unique dataset was meticulously researched and prepared by AI Inventor Emirhan BULUT. It captures valuable information on social media usage and the dominant emotional state of users based on their activities. The dataset is ideal for exploring the relationship between social media usage patterns and emotional well-being. Features:



st.write("Files:")
st.write("train.csv: Data for training models.")
st.write("test.csv: Data for testing models.")
st.write("val.csv: Data for validation purposes.")



st.header(":rainbow[My first application], :+1: :sunglasses:")
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.write(r"$\textsf{\Large Here's our first attempt at using data to create a table}$")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

st.write(r"$\textsf{\Large Random table with 11 row and 21 column:}$")
dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)

st.write(r"$\textsf{\Large Let's expand on the first example using the Pandas Styler object to highlight some elements in the interactive table.}$")
dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

st.write(r"$\textsf{\Large We can use st.table()}$")
dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)

st.write (r"$\textsf{\Large Line Chart}$")
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.write(r"$\textsf{\Large A map}$")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [-20.2, 57.5],
    columns=['lat', 'lon'])

st.map(map_data)

st.write(r"$\textsf{\Large Widgets}$")
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

st.write(r"$\textsf{\Large widgets as key}$")
st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name

st.write(r"$\textsf{\Large Checkbox}$")
st.write("checkbox")
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

st.write(r"$\textsf{\Large Selectbox for option}$")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)
st.write(r"$\textsf{\Large To create a button}$")
left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

st.write(r"$\textsf{\Large Simple the drawing of a chart}$")
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)
