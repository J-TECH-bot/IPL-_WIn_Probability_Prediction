import pipe
from sklearn.pipeline import Pipeline
import streamlit as st
import joblib
import pandas as pd

teams = ['Sunrisers Hyderabad', 'Mumbai Indians',
         'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Mumbai', 'Kolkata', 'Delhi', 'Hyderabad', 'Bangalore', 'Chennai',
          'Jaipur', 'Cape Town', 'Durban', 'Port Elizabeth', 'Centurion',
          'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
          'Raipur', 'Abu Dhabi', 'Sharjah', 'Ranchi']

pipeline = joblib.load('joblib_model.pkl')

st.title('IPL Probability Prediction')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select the city', sorted(cities))
target = st.number_input('Target')
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs')
with col5:
    wickets = st.number_input('Wickets')

if st.button('predict probability'):
    runs_left = target - wickets
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left * 6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [wickets], 'crr': [crr], 'rrr': [rrr]})

    st.table(input_df)
    #features = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'crr', 'rrr']
    # X = input_df[features].to_numpy()
    # st.table(X)

    result = pipe.predict_proba(input_df)
    st.text(result)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "_" + str(round(win*100)) + "%")
    st.header(bowling_team + "_" + str(round(loss*100)) + "%")
