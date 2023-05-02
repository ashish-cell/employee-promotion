import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import sqlite3
import csv
conn = sqlite3.connect('data.db',check_same_thread=False)
cur = conn.cursor()
conn.commit()

st.write('Monitoring App')

st.write('Actual data : ')
SQL_Query1 = pd.read_sql_query(
        '''select
          *
          from result_form''', conn)

df = pd.DataFrame(SQL_Query1, columns=['employee_id','actual_output'])
print(df)
st.write(df)
conn = sqlite3.connect('data.db',check_same_thread=False)
st.write('Database data : ')
SQL_Query2 = pd.read_sql_query(
        '''select
          *
          from app_form''', conn)

df2 = pd.DataFrame(SQL_Query2, columns=['employee_id','department','region','education','gender','recruitment_channel','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met,awards_won','avg_training_score','is_promoted','feedback'])
print(df2)
st.write(df2)

conn.commit()
conn.close()

