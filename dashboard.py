## LIBRAIRIES ################
from ast import literal_eval
import pandas as pd
import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
shap.initjs()
import matplotlib.pyplot as plt #
import warnings #
import json #

# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
# Filter the warning related to is_sparse deprecation
warnings.filterwarnings("ignore", message="is_sparse is deprecated and will be removed in a future version.")


######################################################################""
### PALETTE OF COLORS ######################
rouge ='#88001B'
bleu ='#000064'
vert ='#055D00'
choco = '#4b2312'
magenta = '#FF00FF'


#############################################################################
## WARNINGS MANAGMENT #####
# Suppress FutureWarnings from sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# Suppress UserWarnings from shap
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
# Suppress UserWarnings from streamlit_shap
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit_shap")


##############################################################################
## DECLARING ALL THE FUNCTIONS NEEDED ###
#HOSTING
def host(local:bool):
    if local is True:
        HOST_ = 'http://127.0.0.1:8000'
    else:
        HOST_ = 'https://basicwebappvl.azurewebsites.net/' #Azure 
    return HOST_

HOST = host(local=False)

#GET LIST OF IDS
def get_ids():    
    try:
        response = requests.get(HOST+'/get_ids/')
        response.raise_for_status()  # Check for HTTP errors
        ids = eval(response.content)['data']
        return ids
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")

list_ids = get_ids()

#GET LIST OF FEATURES
def get_feat():    
    try:
        response = requests.get(HOST+'/get_feat/')
        response.raise_for_status()  # Check for HTTP errors
        feat = eval(response.content)['data']
        return feat
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")

features = get_feat()


#GET JSON_CLIENT
def get_json(id:int):    
    try:
        response = requests.get(f'{HOST}/get_json?param_name={id}', timeout=80)
        response.raise_for_status()  # Check for HTTP errors
        json_data = eval(response.content)['data']
        return json_data
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")

#Get the prediction
def get_prediction(id_client: int):
    json_client = get_json(id_client)
    try:
        response = requests.get(HOST+'/prediction/', data=json_client, timeout=80)
        response.raise_for_status()  # Check for HTTP error
        proba_default = eval(response.content)["probability"]
        result = round(proba_default*100, 1)
        return result 
    except requests.RequestException as e:
        print(f"Error fetching prediction from API: {e}")


## Gaugeplot
def gauge(var, var2):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 100-var2,
    domain = {'x': [0, 1], 'y': [0, 1]},
    number = {"prefix": "Score : ", "suffix": " %"},
    title = {'text': f"Probabilité du client {str(var)} d'être solvable (classe 0))"},
    gauge = {'axis': {'range': [None, 100]},
            'threshold' : {'line': {'color': choco, 'width': 4}, 'thickness': 0.75, 'value': 60},
            'bar': {'color': vert if (100-var2) > 60 else rouge}}))
    fig.update_layout(font = {'color': "darkblue", 'family': "Arial"})
    return fig



def kde_fig(id: int, Feature: str):
    json_client = json.loads(get_json(id))
    df = pd.Series(json_client).to_frame().transpose().to_dict()
    #print(df)
    x = df[Feature][0]
    print(x)
    if x == np.nan:
        x = 0
    try:
        response = requests.get(f'{HOST}/get_kde?param_name={Feature}', timeout=80)
        response.raise_for_status()  # Check for HTTP error
        x0 = np.array(eval(response.content)["feat0"])
        x1 = np.array(eval(response.content)["feat1"])
        fig = go.Figure()
        fig.add_trace(go.Violin(x=x0, line_color=rouge, name='Classe 1 (non solvable)', y0=0, opacity=0.4))
        fig.add_trace(go.Violin(x=x1, line_color=bleu, name= 'Classe 0 (solvable)', y0=0, opacity=0.4))

        fig.update_traces(orientation='h', side='positive', meanline_visible=True)
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)   
        # Add title
        fig.update_layout(title_text='Distribution des classes 0 et 1')
        fig.add_vline(x=x, line_width=3, line_dash="dash", line_color=magenta)
        fig.add_annotation(x=x, y=0,
                text=f"Value = {round(x, 2)}",
                showarrow=True,
                font=dict(family="sans serif", size=18, color=magenta),
                arrowhead=2)
        #fig.show()
        return fig
    except requests.RequestException as e:
        print(f"Error fetching prediction from API: {e}")


##shap
def get_shap(id_client: int):
    try:
        response = requests.get(f'{HOST}/get_shap?param_name={id_client}', timeout=80)
        response.raise_for_status()  # Check for HTTP error
        shap_values = eval(response.content)["values"]
        shap_values = np.array(shap_values)
        shap_features = eval(response.content)["feat"]
        shap_features = np.array(shap_features)
        shap_base = eval(response.content)["base"]
        shap_base = np.array(shap_base)
        shap_data = eval(response.content)["data"]
        shap_data = np.array(shap_data)
        explanation = shap.Explanation(values=shap_values, 
                                       feature_names = shap_features, base_values = shap_base,
                                       data = shap_data)
        fig = st_shap(shap.plots.waterfall(explanation[0], max_display=8))
        return fig
    except requests.RequestException as e:
        print(f"Error fetching prediction from API: {e}")
 
###################################################################
## Streamlit CODE ##########
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

##add images + title
col1, col2, col3 = st.columns([1, 7, 1])
with col1:
    st.image('./ressources/Image1.jpg', width=150)

with col2:
    st.markdown("<h1 style='text-align: center;'>Projet 7 : 'Implémentez un modèle de scoring'</h1>", 
                unsafe_allow_html=True)

with col3:
    st.image('./ressources/Image2.png', width=140)

col1, col2 = st.columns([2, 7])
with col1:
    # Selectbox
    ID = st.selectbox('Sélection du client par son numéro ID', list_ids)

with col2:
    st.plotly_chart(gauge(ID, get_prediction(ID)), use_container_width=True)

col1, col2, col3 = st.columns([2, 5, 2])
with col1:
# Selectbox
    feature = st.selectbox('Sélection de la feature', features)

with col2:
    st.write(' ')

with col3:
    st.write(f'SHAP values du client n°{str(ID)}')

col1, col2 = st.columns([1, 1])
with col1:
    st.plotly_chart(kde_fig(ID, feature), use_container_width=True)

with col2:
    get_shap(ID)