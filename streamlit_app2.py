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
cur.execute("""create table if not exists result_form(employee_id text(10),actual_output int);""")
conn.commit()

def addData(employee_id,actual_output):
	# cur.execute("""create table if not exists app_form(employee_id text(10),department text(20),region text(20),education text(20),gender text(1),recruitment_channel text(20),
    # no_of_trainings int, age int, previous_year_rating int,length_of_service int, KPIs_met int,awards_won int, avg_training_score int,is_promoted int,feedback text(5));""")
	# st.write("""create table if not exists clg_form(name text(10),q1 text(10),q2 text(10),q3 text(10),q4 text(10),q5 text(10));""")
	# st.write("INSERT INTO clg_form values"+str((name,a[0],b[0],c[0],d[0],e[0])))
	cur.execute("INSERT INTO result_form values"+str((employee_id,actual_output))+';')
	conn.commit()
	conn.close()
	st.success('Successfully submitted')

with st.form(key='my_form'):
    # Add a text input
    employee_id = st.text_input(label='Enter your employee_id : ')
    actual_output = st.number_input("Enter the actual output:", min_value=0, max_value=1, value=0, step=1)
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    addData(employee_id , actual_output)

conn = sqlite3.connect('data.db',check_same_thread=False)
SQL_Query = pd.read_sql_query(
        '''select
          *
          from result_form''', conn)

df = pd.DataFrame(SQL_Query, columns=['employee_id','actual_output'])
print(df)
st.write(df)

conn.commit()
conn.close()

