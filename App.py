import re
import json
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, callback_context
import dash_table as dt
from dash_table import DataTable
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.client import Config


#### Dash Layer #######################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

app.layout = html.Div([     

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Selected Wells**
                            """),
            
            dbc.Row([dbc.Col(html.Div(id='my-output'), md=5),
                     dbc.Col(html.Div(id='logs'), lg=7)
                    ]),
            html.Div([html.Button("Download Las File", id="btn-download-las", n_clicks=0),
                      html.Div(id='downloaded')
                      #dcc.Download(id="download-las")
                    ]),
            
                   ], 
            ),
  
    ])
])


## Callbacks ##############################################################################################


@app.callback(
    Output("downloaded", "children"),
    Input("btn-download-las", "n_clicks"),
    prevent_initial_call=True,
             )
def load_las(n_clicks):
    return 'Hi Mother Fucker!'#dcx.send_file(path) #dict(content="las", filename=filename)

                   
if __name__ == '__main__':
    app.run_server(port='8080')
