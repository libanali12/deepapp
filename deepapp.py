#!/usr/bin/env python
# coding: utf-8

# In[173]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Nadam
import streamlit as st
import itertools
from io import BytesIO
import requests
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[123]:


@st.experimental_memo
def dataframe():
    url = 'https://raw.githubusercontent.com/libanali12/deepapp/main/heart_2020_cleaned.csv'    
    data = pd.read_csv(url)
    return data


# In[124]:


df=dataframe()


# In[125]:


sc=StandardScaler()


# In[126]:


hdx=df['HeartDisease'].value_counts()
hdy=df['HeartDisease'].value_counts().index


# In[127]:


df1=df[df["HeartDisease"] == "Yes"]
smokex=df1['Smoking'].value_counts()
smokey=df1['Smoking'].value_counts().index


# In[128]:


Alcoholx=df1['AlcoholDrinking'].value_counts()
Alcoholy=df1['AlcoholDrinking'].value_counts().index


# In[129]:


sexx=df1['Sex'].value_counts()
sexy=df1['Sex'].value_counts().index


# In[130]:


agex=df1['AgeCategory'].value_counts()
agey=df1['AgeCategory'].value_counts().index


# In[131]:


racex=df1['Race'].value_counts()
racey=df1['Race'].value_counts().index


# In[132]:


physicalx=df1['PhysicalActivity'].value_counts()
physicaly=df1['PhysicalActivity'].value_counts().index


# In[133]:


asthmax=df1['Asthma'].value_counts()
asthmay=df1['Asthma'].value_counts().index


# In[134]:


kidneyx=df1['KidneyDisease'].value_counts()
kidneyy=df1['KidneyDisease'].value_counts().index


# In[135]:


skinx=df1['SkinCancer'].value_counts()
skiny=df1['SkinCancer'].value_counts().index


# In[136]:


diabeticx=df1['Diabetic'].value_counts()
diabeticy=df1['Diabetic'].value_counts().index


# In[137]:


strokex=df1['Stroke'].value_counts()
strokey=df1['Stroke'].value_counts().index


# In[138]:


bmix=df1['BMI'].value_counts()
bmiy=df1['BMI'].value_counts().index


# In[139]:


df2=df[df["HeartDisease"] == "No"]
smokex1=df2['Smoking'].value_counts()
smokey1=df2['Smoking'].value_counts().index


# In[140]:


Alcoholx1=df2['AlcoholDrinking'].value_counts()
Alcoholy1=df2['AlcoholDrinking'].value_counts().index


# In[141]:


sexx1=df2['Sex'].value_counts()
sexy1=df2['Sex'].value_counts().index


# In[142]:


agex1=df2['AgeCategory'].value_counts()
agey1=df2['AgeCategory'].value_counts().index


# In[143]:


racex1=df2['Race'].value_counts()
racey1=df2['Race'].value_counts().index


# In[144]:


physicalx1=df2['PhysicalActivity'].value_counts()
physicaly1=df2['PhysicalActivity'].value_counts().index


# In[145]:


asthmax1=df2['Asthma'].value_counts()
asthmay1=df2['Asthma'].value_counts().index


# In[146]:


kidneyx1=df2['KidneyDisease'].value_counts()
kidneyy1=df2['KidneyDisease'].value_counts().index


# In[147]:


skinx1=df2['SkinCancer'].value_counts()
skiny1=df2['SkinCancer'].value_counts().index


# In[148]:


diabeticx1=df2['Diabetic'].value_counts()
diabeticy1=df2['Diabetic'].value_counts().index


# In[149]:


strokex1=df2['Stroke'].value_counts()
strokey1=df2['Stroke'].value_counts().index


# In[150]:


bmix1=df2['BMI'].value_counts()
bmiy1=df2['BMI'].value_counts().index


# In[151]:


@st.experimental_memo
def load_model():
    link = "https://raw.githubusercontent.com/libanali12/deepapp/main/model.pkl"
    mfile = BytesIO(requests.get(link).content)
    model= pickle.load(mfile)
    return model


# In[152]:


model = load_model()


# In[153]:


colours = {"Yes": "#0C3B5D","No": "#51CD3E",}


# In[154]:


heart_df = pd.get_dummies(df,prefix=None,drop_first=True)
y = heart_df['HeartDisease_Yes']
x = heart_df
x = x.drop('HeartDisease_Yes',axis=1)


# In[192]:


x=x.astype('float')


# In[200]:


def main():
    st.title("Heart Disease Classification App")
    menu = ["Home","Deep Learning Model"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        st.write('Heart Disease Dataset')
        st.dataframe(df.head(5))
        st.subheader("Exploratory Data Analysis")
        option = st.selectbox('Select EDA Below',("People with Heart Disease",'People That Dont Have Heart Disease'))
        if option == "People with Heart Disease":
            fig1 = px.bar(df1,x=smokey, y=smokex,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig1.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig1.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig1.update_layout(showlegend=False, title="People with Heart Disease that smoke yes/no",
                  xaxis_title="Does the person smoke yes/no",
                  yaxis_title="Count")
            fig1.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig1.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig1)
            fig2 = px.bar(df1,x=Alcoholy, y=Alcoholx,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig2.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig2.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig2.update_layout(showlegend=False, title="People with Heart Disease that smoke yes/no",
                  xaxis_title="Does the person drink yes/no",
                  yaxis_title="Count")
            fig2.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig2.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig2)
            fig3 = px.bar(df1,x=sexy, y=sexx,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig3.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig3.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig3.update_layout(showlegend=False, title="People with Heart Disease Gender Distribution",
                  xaxis_title="Gender Distribution",
                  yaxis_title="Count")
            fig3.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig3.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig3)
            fig4 = px.bar(df1,x=agex, y=agey,
             color_discrete_map=colours,
             height=600, width=900)
            fig4.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig4.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig4.update_layout(showlegend=False, title="People with Heart Disease Age Distribution",
                  xaxis_title="Age Distribution",
                  yaxis_title="Count")
            fig4.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig4.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig4)
            fig5 = px.bar(df1,x=racex, y=racey,
             color_discrete_map=colours,
             height=600, width=900)
            fig5.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig5.update_traces( marker_line_color='#08306b',
                  marker_line_width=2, opacity=0.6)
            fig5.update_layout(showlegend=False, title="People with Heart Disease Race Distribution",
                  xaxis_title="Race Distribution",
                  yaxis_title="Count")
            fig5.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig5.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig5)
            fig6 = px.bar(df1,x=physicaly, y=physicalx,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig6.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig6.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig6.update_layout(showlegend=False, title="People with Heart Disease do Physical Activity yes/no distribution",
                  xaxis_title="Physical Activity yes/no",
                  yaxis_title="Count")
            fig6.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig6.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig6)
            fig7= px.bar(df1,x=asthmay, y=asthmax,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig7.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig7.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig7.update_layout(showlegend=False, title="People with Heart Disease have Asthma yes/no distribution",
                  xaxis_title="Asthma yes/no",
                  yaxis_title="Count")
            fig7.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig7.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig7)
            fig8 = px.bar(df1,x=skiny, y=skinx,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig8.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig8.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig8.update_layout(showlegend=False, title="People with Heart Disease have Skin Cancer yes/no distribution",
                  xaxis_title="Skin Cancer yes/no",
                  yaxis_title="Count")
            fig8.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig8.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig8)
            fig9 = px.bar(df1,x=diabeticy, y=diabeticx,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig9.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig9.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig9.update_layout(showlegend=False, title="People with Heart Disease have Diabetic yes/no distribution",
                  xaxis_title="Diabetic yes/no",
                  yaxis_title="Count")
            fig9.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig9.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig9)
            fig10 = px.bar(df1,x=strokey, y=strokex,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig10.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig10.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig10.update_layout(showlegend=False, title="People with Heart Disease have Stroke yes/no distribution",
                  xaxis_title="Stroke yes/no",
                  yaxis_title="Count")
            fig10.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig10.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig10)
            fig11 = px.histogram(df1,x=bmix, y=bmiy,nbins=50,
             color_discrete_map=colours,
             height=600, width=900)
            fig11.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig11.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig11.update_layout(showlegend=False, title="BMI distribution of People with Heart Disease",
                  xaxis_title="BMI distribution",
                  yaxis_title="Count")
            fig11.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig11.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig11)
        else:
            fig12 = px.bar(df2,x=smokey1, y=smokex1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig12.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig12.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig12.update_layout(showlegend=False, title="People with no Heart Disease that smoke yes/no",
                  xaxis_title="Does the person smoke yes/no",
                  yaxis_title="Count")
            fig12.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig12.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig12)
            fig13 = px.bar(df2,x=Alcoholy1, y=Alcoholx1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig13.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig13.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig13.update_layout(showlegend=False, title="People with no Heart Disease that smoke yes/no",
                  xaxis_title="Does the person drink yes/no",
                  yaxis_title="Count")
            fig13.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig13.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig13)
            fig14 = px.bar(df2,x=sexy1, y=sexx1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig14.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig14.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig14.update_layout(showlegend=False, title="People with no Heart Disease Gender Distribution",
                  xaxis_title="Gender Distribution",
                  yaxis_title="Count")
            fig14.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig14.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig14)
            fig15 = px.bar(df2,x=agex1, y=agey1,
             color_discrete_map=colours,
             height=600, width=900)
            fig15.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig15.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig15.update_layout(showlegend=False, title="People with no Heart Disease Age Distribution",
                  xaxis_title="Age Distribution",
                  yaxis_title="Count")
            fig15.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig15.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig15)
            fig16 = px.bar(df2,x=racex1, y=racey1,
             color_discrete_map=colours,
             height=600, width=900)
            fig16.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig16.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig16.update_layout(showlegend=False, title="People with no Heart Disease Race Distribution",
                  xaxis_title="Race Distribution",
                  yaxis_title="Count")
            fig16.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig16.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig16)
            fig17 = px.bar(df2,x=physicaly1, y=physicalx1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig17.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig17.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig17.update_layout(showlegend=False, title="People with no Heart Disease do Physical Activity yes/no distribution",
                  xaxis_title="Physical Activity yes/no",
                  yaxis_title="Count")
            fig17.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig17.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig17)
            fig18= px.bar(df2,x=asthmay1, y=asthmax1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig18.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig18.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig18.update_layout(showlegend=False, title="People with no Heart Disease have Asthma yes/no distribution",
                  xaxis_title="Asthma yes/no",
                  yaxis_title="Count")
            fig18.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig18.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig18)
            fig19 = px.bar(df2,x=skiny1, y=skinx1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig19.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig19.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig19.update_layout(showlegend=False, title="People with no Heart Disease have Skin Cancer yes/no distribution",
                  xaxis_title="Skin Cancer yes/no",
                  yaxis_title="Count")
            fig19.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig19.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig19)
            fig20 = px.bar(df2,x=diabeticy1, y=diabeticx1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig20.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig20.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig20.update_layout(showlegend=False, title="People with no Heart Disease have Diabetic yes/no distribution",
                  xaxis_title="Diabetic yes/no",
                  yaxis_title="Count")
            fig20.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig20.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig20)
            fig21 = px.bar(df2,x=strokey1, y=strokex1,orientation='v',
             color_discrete_map=colours,
             height=600, width=900)
            fig21.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig21.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig21.update_layout(showlegend=False, title="People with no Heart Disease have Stroke yes/no distribution",
                  xaxis_title="Stroke yes/no",
                  yaxis_title="Count")
            fig21.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig21.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig21)
            fig22 = px.histogram(df2,x=bmix1, y=bmiy1,nbins=50,
             color_discrete_map=colours,
             height=600, width=900)
            fig22.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
            fig22.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
            fig22.update_layout(showlegend=False, title="BMI distribution of People with no Heart Disease",
                  xaxis_title="BMI distribution",
                  yaxis_title="Count")
            fig22.update_xaxes(showline=True, linewidth=1, linecolor='black')
            fig22.update_yaxes(showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig22)
    else:
        st.subheader("Deep Learning Model")
        st.sidebar.header('Specify Input Parameters')
        BMI = st.sidebar.slider('BMI', min_value=float(x.BMI.min()),max_value=float(x.BMI.max()),step=float(x.BMI.mean()))
        PhysicalHealth = st.sidebar.slider('PhysicalHealth',min_value=float(x.PhysicalHealth.min()), max_value=float(x.PhysicalHealth.max()), step=float(x.PhysicalHealth.mean()))
        MentalHealth = st.sidebar.slider('MentalHealth',min_value= float(x.MentalHealth.min()),max_value= float(x.MentalHealth.max()),step=float(x.MentalHealth.mean()))
        SleepTime = st.sidebar.slider('SleepTime', min_value=float(x.SleepTime.min()),max_value= float(x.SleepTime.max()),step=float(x.SleepTime.mean()))
        Smoking = st.sidebar.selectbox('Smoking',(0,1))
        AlcoholDrinking= st.sidebar.selectbox('AlcoholDrinking',(0,1))
        Stroke = st.sidebar.selectbox('Stroke',(0,1))
        DiffWalking = st.sidebar.selectbox('DiffWalking',(0,1))
        Sex = st.sidebar.selectbox('Sex', (0,1))
        AgeCategory_25_29 = st.sidebar.selectbox('AgeCategory_25-29',(0,1))
        AgeCategory_30_34 = st.sidebar.selectbox('AgeCategory_30-34',(0,1))
        AgeCategory_35_39 = st.sidebar.selectbox('AgeCategory_35-39', (0,1))
        AgeCategory_40_44 = st.sidebar.selectbox('AgeCategory_40-44',(0,1))
        AgeCategory_45_49 = st.sidebar.selectbox('AgeCategory_45-49', (0,1))
        AgeCategory_50_54 = st.sidebar.selectbox('AgeCategory_50-54', (0,1))
        AgeCategory_55_59 = st.sidebar.selectbox('AgeCategory_55-59', (0,1))
        AgeCategory_60_64 = st.sidebar.selectbox('AgeCategory_60-64', (0,1))
        AgeCategory_65_69 = st.sidebar.selectbox('AgeCategory_65-69', (0,1))
        AgeCategory_70_74 = st.sidebar.selectbox('AgeCategory_70-74', (0,1))
        AgeCategory_75_79 =st.sidebar.selectbox('AgeCategory_75-79',(0,1))
        AgeCategory_80_older= st.sidebar.selectbox('AgeCategory_80 or older', (0,1))
        Race_Asian = st.sidebar.selectbox('Race_Asian', (0,1))
        Race_Black = st.sidebar.selectbox('Race_Black', (0,1))
        Race_Hispanic = st.sidebar.selectbox('Race_Hispanic', (0,1))
        Race_Other = st.sidebar.selectbox('Race_Other',(0,1))
        Race_White = st.sidebar.selectbox('Race_White',(0,1))
        Diabetic_No_borderline_diabetes = st.sidebar.selectbox('Diabetic_No, borderline diabetes',(0,1))
        Diabetic_Yes = st.sidebar.selectbox('Diabetic_Yes', (0,1))
        Diabetic_Yes_during_pregnancy = st.sidebar.selectbox('Diabetic_Yes (during pregnancy)',(0,1))
        PhysicalActivity_Yes = st.sidebar.selectbox('PhysicalActivity_Yes',(0,1))
        GenHealth_Fair = st.sidebar.selectbox('GenHealth_Fair', (0,1))
        GenHealth_Good = st.sidebar.selectbox('GenHealth_Good', (0,1))
        GenHealth_Poor = st.sidebar.selectbox('GenHealth_Poor',(0,1))
        GenHealth_Very_good = st.sidebar.selectbox('GenHealth_Very good', (0,1))
        Asthma = st.sidebar.selectbox('Asthma_Yes', (0,1))
        KidneyDisease = st.sidebar.selectbox('KidneyDisease_Yes', (0,1))
        SkinCancer= st.sidebar.selectbox('SkinCancer_Yes', (0,1))       
        data = {'BMI': BMI, 'PhysicalHealth':PhysicalHealth,'MentalHealth':MentalHealth,'SleepTime':SleepTime,
            'Smoking_Yes':Smoking,'AlcoholDrinking_Yes': AlcoholDrinking,'Stroke_Yes':Stroke,'DiffWalking_Yes':DiffWalking,
            'Sex_Male':Sex,'AgeCategory_25-29':AgeCategory_25_29,'AgeCategory_30-34':AgeCategory_30_34,'AgeCategory_35-39':AgeCategory_35_39,
            'AgeCategory_40-44':AgeCategory_40_44,'AgeCategory_45-49':AgeCategory_45_49,'AgeCategory_50-54':AgeCategory_50_54,
            'AgeCategory_55-59':AgeCategory_55_59,'AgeCategory_60-64':AgeCategory_60_64,'AgeCategory_65-69': AgeCategory_65_69,
            'AgeCategory_70-74':AgeCategory_70_74,'AgeCategory_75-79':AgeCategory_75_79,'AgeCategory_80 or older':AgeCategory_80_older,
            'Race_Asian': Race_Asian,'Race_Black':Race_Black,'Race_Hispanic':Race_Hispanic,'Race_Other':Race_Other,
            'Race_White':Race_White,'Diabetic_No, borderline diabetes':Diabetic_No_borderline_diabetes,'Diabetic_Yes':Diabetic_Yes,
            'Diabetic_Yes (during pregnancy)':Diabetic_Yes_during_pregnancy, 'PhysicalActivity_Yes':PhysicalActivity_Yes,
            'GenHealth_Fair': GenHealth_Fair,'GenHealth_Good':GenHealth_Good,'GenHealth_Poor':GenHealth_Poor,'GenHealth_Very good':GenHealth_Very_good,
            'Asthma_Yes':Asthma,'KidneyDisease_Yes':KidneyDisease,'SkinCancer_Yes':SkinCancer}
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a Deep Learning model was constructed using survey data of over 300k US residents from the year 2020.
        
        To predict your heart disease status, simply follow the steps bellow:
        1. For the Input features with a Max of 1 and Min of 0 this is Yes and No parameters Yes meaning 1 and No meaning 0. Also for the Sex parameter Female is 0 and Male is 1.For physical health and mental health parameters measure how many days out of the month were you physical and mental good.Finally for the sleep parameter it measure how much sleep you get in 24 hr period.Enter the parameters that best describe you.
        2. Press the "Predict" button and wait for the result.""")
        if st.button("Predict Diagnosis"):
            df4= pd.DataFrame(data, index=[0])
            result= (model.predict(df4) > 0.7).astype(int)
            probability = model.predict(df4)
            result1= itertools.chain.from_iterable(result)
            probability1= itertools.chain.from_iterable(probability)
            resultdf = pd.DataFrame({'Result':result1,'Probability':probability1},index=['Diagnosis'])
            resultdf['Diagnosis'] = ['No Heart Disease' if x == 0 else 'Yes Heart Disease'for x in resultdf['Result']]
            st.write('Diagnosis based on Input Parameters')
            st.dataframe(resultdf)
            
if __name__ == '__main__':
    main() 


# In[ ]:




