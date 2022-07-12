import streamlit as st
import numpy as np
import pandas as pd 
import altair as alt

from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Heart Failure EDA')

st.title('Douglas Crespo')
st.subheader('Heart Failure Prediction using ***Random Forest Classifier***')
st.write('---')

st.sidebar.header('Directory')
app = st.sidebar.selectbox('', ['Explore Data', 'Predict Mortality', 'Citation'])

df = pd.read_csv('cardio_train.csv')

if app == 'Explore Data':
    about_expander = st.beta_expander('About',expanded=True)
    with about_expander:
        img = Image.open('heartattack.jpg')
        st.image(img)
        st.write("""
                Cardiovascular diseases (CVDs) are the **number 1 cause of death** globally, 
                taking an estimated 17.9 million lives each year, which accounts for 31 
                percent of all deaths worlwide. Heart failure is a common event caused 
                by CVDs and this dataset contains 12 features that can be used to predict 
                mortality by heart failure.
                """)

    st.subheader('**Explore the dataset**')
    col1, col2 = st.beta_columns(2)
    selectbox_options = col1.selectbox('Transform', ['Head','Tail', 
                                                        'Describe','Shape', 
                                                        'DTypes', 'Value Count'])
    if selectbox_options == 'Head':
        input_count = col2.number_input('Count', 5, 50, help='min=5, max=50')
        st.write(df.head(input_count))
    elif selectbox_options == 'Tail':
        input_count = col2.number_input('Count', 5, 50, help='min=5, max=50')
        st.write(df.tail(input_count))
    elif selectbox_options == 'Describe':
        st.write(df.describe())
    elif selectbox_options == 'Shape':
        st.write(df.head())
        st.write('Shape: ', df.shape)
    elif selectbox_options == 'DTypes':
        st.write(df.dtypes)
    
    st.write('---')
    numeric_df = df.select_dtypes(['float64', 'int64'])
    numeric_cols = numeric_df.columns

    st.subheader('**Filter columns with Multiselect**')
    st.write("""This feature is for comparing certain columns in the dataset.
                You may add only the columns you wish to compare and explore.
                """)
    feature_selection = st.multiselect('', options=numeric_cols)
    df_features = df[feature_selection]
    st.write(df_features)
    st.write('---')

    st.sidebar.subheader('Visualization Settings')
    y_axis = st.sidebar.selectbox('Select y-axis', ['age', 'ap_hi', 
                                                    'alco'])
    x_axis = st.sidebar.selectbox('Select x-axis', ['ap_lo'])
    
    label = st.sidebar.selectbox('Select label', ['cardio', 'active', 'gluc', 
                                                    'gluc', 'sex', 
                                                    'smoking'])
    st.subheader('**Visualization**')
    st.write("""Customize the x and y axis through the sidebar visualization settings. 
                You can also select binary features as labels which will be in the form 
                of a color.""")
    select_graph = st.sidebar.radio('Select Graph', ('point', 'bar', 'area', 'line'))

    col1, col2, col3 = st.beta_columns([.5,.5,1])
    graph_hgt = col1.slider('Height', 200, 600, 400, step=10)
    graph_wgt = col2.slider('Width',400, 800, 600, step=10)
        
    df = df.loc[(df.creatinine_phosphokinase < 800) & (df.platelets < 500000) & 
                (df.serum_creatinine < 2.2) & (df.age >= 40)]

    chart = alt.Chart(data=df, mark=select_graph).encode(alt.X(x_axis, scale=alt.Scale(zero=False)), 
                                                            alt.Y(y_axis, scale=alt.Scale(zero=False)),color=label).properties(
        height=graph_hgt,width=graph_wgt)
    st.write(chart)        

elif app == 'Predict Mortality':
    st.sidebar.subheader('User Input Features')

    df = pd.read_csv('cardio_train.csv')
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

    def user_input_features():
        display = ("Female", "Male")
        options = list(range(len(display)))
        sex = st.sidebar.radio("Sex", options, format_func=lambda x: display[x])

        smoking = st.sidebar.checkbox('Smoking')
        if smoking:
            smoking = 1
        cholesterol = st.sidebar.checkbox('Cholesterol')
        if cholesterol:
            cholesterol = 1
        gluc = st.sidebar.checkbox('Glucose')
        if gluc:
            gluc = 1
        active = st.sidebar.checkbox('Active')
        if active:
            active = 1
        alco = st.sidebar.checkbox('Alcohol')
        if alco:
            alco = 1
            
        age = st.sidebar.slider('Age', 25, 65, 45)
        ap_hi = st.sidebar.slider('BP Sistolico', 40, 240, 128)
        ap_lo = st.sidebar.slider('Serum Sodium', 113, 148, 130)

        data = {'age': age,
                'active': active,
                'gluc': gluc,
                'ap_hi': ap_hi,
                'gluc': gluc,
                'ap_lo': ap_lo,
                'sex': sex,
                'smoking': smoking,
                'alco': alco
                }
        features = pd.DataFrame(data, index=[0])
        return features

    user_data = user_input_features()
    st.subheader('**User Input parameters**')
    st.write(user_data)
    my_expander = st.beta_expander('Check dataset')
    with my_expander:
        st.write(df)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    user_result = classifier.predict(user_data)

    st.title('')
    st.subheader('**Conclusion:**')
    pred_button = st.button('Predict')
    if pred_button:
        if user_result[0] == 0:
            st.success('Patient without risk of heart disease')
        else:
            st.error('Patient at risk of heart disease')

else:
    st.header('**References/Citation**')
