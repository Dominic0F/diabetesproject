import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from flask import Flask
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import sys

# Fix potential encoding issues with logging
sys.stdout.reconfigure(encoding='utf-8')

# Load the dataset
file_path = 'Resources/final_data.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# Check if the model file exists
model_path = 'Resources/diabetes_prevalence_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app with suppress_callback_exceptions
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Standardize the features
features = ['Daily Caloric Supply', 'Animal Protein Supply', 'Vegetal Protein Supply']
scaler = StandardScaler()
scaler.fit(data[features])

# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label='Dashboard', tab_id='tab-dashboard'),
        dbc.Tab(label='Predict Diabetes Prevalence', tab_id='tab-predict')
    ], id='tabs', active_tab='tab-dashboard'),
    html.Div(id='tab-content')
], fluid=True)

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab_content(active_tab):
    if active_tab == 'tab-dashboard':
        return dbc.Row([
            dbc.Col([
                html.H1("Diabetes Prevalence Dashboard"),
                html.H3("Generate Pie Chart"),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in data['Country'].unique()],
                    value=data['Country'].unique()[0]
                ),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[],
                    value=None
                ),
                dcc.Graph(id='pie-chart'),
                html.Div(id='diabetes-prevalence', style={'fontSize': 24, 'textAlign': 'center'})
            ], width=12)
        ])
    elif active_tab == 'tab-predict':
        return dbc.Row([
            dbc.Col([
                html.H1("Predict Diabetes Prevalence"),
                html.Label('Daily Caloric Supply'),
                dcc.Input(id='caloric-supply', type='number', value=0, min=0),
                html.Label('Animal Protein Supply (grams)'),
                dcc.Input(id='animal-protein', type='number', value=0, min=0),
                html.Label('Vegetal Protein Supply (grams)'),
                dcc.Input(id='vegetal-protein', type='number', value=0, min=0),
                html.Br(),
                html.Button('Predict', id='predict-button', n_clicks=0),
                html.Div(id='prediction-output', style={'fontSize': 24, 'textAlign': 'center', 'marginTop': '20px'})
            ], width=6)
        ])

@app.callback(
    Output('year-dropdown', 'options'),
    Output('year-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def set_year_options(selected_country):
    filtered_data = data[data['Country'] == selected_country]
    year_options = [{'label': year, 'value': year} for year in filtered_data['Year'].unique()]
    return year_options, year_options[0]['value']

@app.callback(
    Output('pie-chart', 'figure'),
    Output('diabetes-prevalence', 'children'),
    Input('country-dropdown', 'value'),
    Input('year-dropdown', 'value')
)
def update_pie_chart(selected_country, selected_year):
    try:
        filtered_data = data[(data['Country'] == selected_country) & (data['Year'] == selected_year)].iloc[0]
        
        # Calculate the slices for the pie chart
        animal_protein_calories = filtered_data['Animal Protein Supply'] * 5
        vegetal_protein_calories = filtered_data['Vegetal Protein Supply'] * 5
        other_caloric_supply = filtered_data['Daily Caloric Supply'] - (animal_protein_calories + vegetal_protein_calories)
        
        # Create pie chart
        fig = px.pie(
            names=['Animal Protein Calories', 'Plant Protein Calories', 'Other'],
            values=[animal_protein_calories, vegetal_protein_calories, other_caloric_supply],
            title='Protein and Caloric Supply Breakdown'
        )
        
        # Update layout to make the pie chart larger
        fig.update_layout(
            width=800,  # Adjust the width as needed
            height=800  # Adjust the height as needed
        )
        
        # Display diabetes prevalence
        diabetes_prevalence_text = f"Diabetes Prevalence: {filtered_data['Diabetes Prevalence']:.2f}%"
        
        return fig, diabetes_prevalence_text
    except Exception as e:
        print("Error in pie chart callback:", e)
        return {}, ""

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('caloric-supply', 'value'),
    State('animal-protein', 'value'),
    State('vegetal-protein', 'value')
)
def predict_diabetes_prevalence(n_clicks, caloric_supply, animal_protein, vegetal_protein):
    if n_clicks > 0:
        print(f"Inputs - Caloric Supply: {caloric_supply}, Animal Protein: {animal_protein}, Vegetal Protein: {vegetal_protein}")
        input_data = pd.DataFrame([[caloric_supply, animal_protein, vegetal_protein]], 
                                  columns=['Daily Caloric Supply', 'Animal Protein Supply', 'Vegetal Protein Supply'])
        input_data_scaled = scaler.transform(input_data)
        print(f"Scaled Inputs: {input_data_scaled}")
        prediction = model.predict(input_data_scaled)
        print(f"Prediction: {prediction}")
        return f"Predicted Diabetes Prevalence: {prediction[0][0]:.2f}%"
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
