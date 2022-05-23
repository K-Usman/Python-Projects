import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,r2_score
import os

st.title('ML App for Classification and Regression')
st.info('Please upload the csv file')
upFile=st.file_uploader('')
disableBool=True
try:
        if upFile is not None:
                df=pd.read_csv(upFile,encoding='unicode_escape')
                disableBool=False
                st.write(df)
                columnNames=df.columns
                indVariables=st.multiselect('Please Select independent variables',columnNames)
                depVariable=st.selectbox('Please Select dependent variables',columnNames)
                X=df[indVariables]
                y=df[depVariable]
                testSize=st.slider('Please specify the test dataset size',0.,1.,0.1)
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=testSize) 
                modelCategory=st.selectbox('Please select the model category',('Regression','Classification'))
                if modelCategory=='Regression':
                    modelSelection=st.selectbox('Please select the regression model',('Multiple Linear Regression','Support Vector Regression','Random Forest Regression'))
                else:
                    modelSelection=st.selectbox('Please select the classification model',('Decision Tree','Naive Bayes','Random Forest'))    
        else:
                indVariables=st.multiselect('Please Select independent variables',[''])
                depVariable=st.selectbox('Please Select dependent variables',[''])
                testSize=st.slider('Please specify the test dataset size',0.,1.,0.1)
                modelCategory=st.selectbox('Please select the model category',('Regression','Classification'))
                modelSelection=st.selectbox('Please select the model',[''])      
except UnicodeDecodeError:
    st.error('This file format is not supported')
except ValueError:
    st.error('Test dataset size cannot be 0')

def buildModel():
    if modelCategory=='Regression':
        if modelSelection=='Multiple Linear Regression':
            modelMLR()
        elif modelSelection=='Support Vector Regression':
            modelSVR()
        else:
            modelRFR()
    elif modelCategory=='Classification':
        if modelSelection=='Decision Tree':
            modelDT()
        elif modelSelection=='Naive Bayes':
            modelNB()
        else:
            modelRF()

def modelDT():
    model=DecisionTreeClassifier()
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    prDF=pd.DataFrame(predictions,y_test)
    accuracy=accuracy_score(predictions,y_test)
    st.info(f'The accuracy score for this mode is: {accuracy}')
    return st.dataframe(prDF)
def modelRF():
    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    prDF=pd.DataFrame(predictions,y_test)
    accuracy=accuracy_score(predictions,y_test)
    st.info(f'The accuracy score for this mode is: {accuracy}')
    return st.table(prDF)
def modelNB():
    model=GaussianNB()
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    prDF=pd.DataFrame(predictions,y_test)
    accuracy=accuracy_score(predictions,y_test)
    st.info(f'The accuracy score for this mode is: {accuracy}')
    return st.dataframe(prDF)
def modelMLR():
    model=LinearRegression()
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    prDF=pd.DataFrame(predictions,y_test)
    rsq=r2_score(y_test,predictions)
    st.info(f'The rsquare for this model is: {rsq}')
    return st.dataframe(prDF)
def modelSVR():
    model=SVR()
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    prDF=pd.DataFrame(predictions,y_test)
    rsq=r2_score(y_test,predictions)
    st.info(f'The rsquare for this model is: {rsq}')
    return st.dataframe(prDF)
def modelRFR():
    model=RandomForestRegressor()
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    prDF=pd.DataFrame(predictions,y_test)
    rsq=r2_score(y_test,predictions)
    st.info(f'The rsquare for this model is: {rsq}')
    return st.dataframe(prDF)

if upFile is None:
    state=True
elif disableBool==True:
    state=True
else:
    state=False
if st.button('Submit',disabled=state):
    try:
        buildModel()
    except ValueError:
        st.error("Please enter the correct data")



