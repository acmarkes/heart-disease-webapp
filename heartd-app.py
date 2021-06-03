#%%
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart Disease Prediction App Demo
This app predicts the chances of a person having a heart disease.
Data obtained from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

*Warning:* This app is in no way adequate for actual medical diagnosis 
and serves only as a simple demo of Streamlit

""")

st.sidebar.header('User Input Features')

def user_input_features():
    sex = st.sidebar.selectbox('Sex',('Male','Female'))
    age = st.sidebar.slider('Age', min_value=16, max_value=110)
    thal = st.sidebar.selectbox('Defects',('normal','fixed defects','reversable defect'))
    ca = st.sidebar.slider('Number of major vessels (0-3) colored by flourosopy', min_value=0, max_value=3)
    cp = st.sidebar.selectbox('Chest pain',('typical angina','atypical angina','non-anginal pain','asymptomatic'))

    data = {'sex': sex,
            'age': age,
            'thal': thal,
            'ca': ca,
            'cp': cp}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

#Building a small dataset to make encoding possible

col_names = ['age', 'sex',
             'cp','trestbps',
             'chol','fbs',
             'restecg','thalach','exang',
             'oldpeak','slope',
             'ca','thal','num']

heartd_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                        index_col = False, names=col_names)

heartd = heartd_raw.drop(columns=['num'])

#Join the user input to the dataframe
df = pd.concat([input_df,heartd.iloc[0:100,:]],axis=0)

#Filling the null values with 0
df.fillna(0,inplace=True)

# Display the user input features
st.subheader('User Input features')
st.dataframe(df.iloc[:1,:5])

#Replace values
df['sex'].replace({'Male':1,'Female':0},inplace=True)
df['thal'].replace({'normal':3, 'fixed defect':6, 'reversable defect':7},inplace=True)
df['cp'].replace({'typical angina':1,'atypical angina':2,'non-anginal pain':3,'asymptomatic':4},inplace=True)


#Encode the features
encode = ['thal','cp']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1] # Selects only the first row (the user input data)
df['error_fix'] = 0

# Reads in saved classification model
load_clf = pickle.load(open('heartd_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
diagnosis = np.array(['< 50% diameter narrowing','> 50% diameter narrowing'])
st.write(diagnosis[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)