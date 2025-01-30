import streamlit as st
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from fn import *

def drop(df, col):
    df.drop(col, axis=1, inplace=True)

def ugsys(df):
    titles = list(df.columns)
    questions = titles[1:]
    for q in questions:
        df.loc[df[q] == 3, q] = 0
        if q == 'Self Esteem Q2':
            df.loc[df[q] == 1, q] = 5
            df.loc[df[q] == 2, q] = 3
            df.loc[df[q] == 4, q] = -2
            df.loc[df[q] == 5, q] = -4
        else:
            df.loc[df[q] == 1, q] = -1
            df.loc[df[q] == 2, q] = 0
            df.loc[df[q] == 4, q] = 3
    return df

def merge(df):
    ADHD = ['ADHD Q1', 'ADHD Q2', 'ADHD Q3', 'ADHD Q4']
    df['ADHD Score'] = df[ADHD].sum(axis=1)
    Anxiety = ['Anxiety Q1', 'Anxiety Q2']
    df['Anxiety Score'] = df[Anxiety].sum(axis=1)
    SelfEsteem = ['Self Esteem Q1', 'Self Esteem Q2','Self Esteem Q3']
    df['Self Esteem Score'] = df[SelfEsteem].sum(axis=1)
    Depression = ['Depression Q1', 'Depression Q2','Depression Q3']
    df['Depression Score'] = df[Depression].sum(axis=1)
    Total = ['ADHD Score', 'Anxiety Score','Self Esteem Score','Depression Score']
    df['Total Score'] = df[Total].sum(axis=1)
    drop(df, df.iloc[:, 1:13])
    return df

def refine(df):
    df = ugsys(df)
    df = merge(df)
    drop(df, 'Total Score')
    return df
    
def predict(df):
    model = joblib.load("model.pkl")
    prediction = model.predict(df)
    return prediction[0]
