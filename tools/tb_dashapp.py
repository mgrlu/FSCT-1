import os
import time
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

infile = r"C:\Users\gilcher\code\STEMDETECTION\code\FSCT\model\training_history.csv"
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-plot'),
    dcc.Interval(
        id='interval-component',
        interval=10*1000,  # 10 seconds in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('live-plot', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    df = pd.read_csv(infile, header=None, index_col=False, sep=' ')
    df.columns = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
    fig = go.Figure()

    for col in ['loss', 'accuracy']:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))

    fig.update_layout(
        title='Live CSV Data',
        xaxis_title='Time',
        yaxis_title='Value',
        legend_title='Columns'
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
