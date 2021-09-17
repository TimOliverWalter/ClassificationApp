import pandas as pd
import base64
import io

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import dash_daq as daq

import plotly.express as px
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

page2 = html.Div([
    html.H5(
        "Text"
    )
])
external_scripts = ["/assets/style.css"]
app = dash.Dash()

app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    id="page-one",
                    label="New Data",
                    children=[
                        html.Div(
                            [
                                dcc.Upload(
                                    id="upload-page-one",
                                    children=html.Div(
                                        [
                                            "Drag and Drop or ",
                                            html.A(
                                                html.B(
                                                    html.I(
                                                        "Select your Excel File"
                                                    )
                                                )
                                            )
                                        ]
                                    ),
                                    style={
                                        "width": "99%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "auto",
                                        "margin-top": "10px",
                                        "margin-bottom": "10px"
                                    }
                                ),
                                html.Div(
                                    id="output-page-one",
                                    children=[]
                                )
                            ]
                        )
                    ]
                ),
                dcc.Tab(
                    id="page-two",
                    label="test",
                    children=[
                        page2
                    ]
                )
            ]
        )
    ]
)


@app.callback(
    Output("output-page-one", "children"),
    Input("upload-page-one", "contents")
)
def update_page_one(contents):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        df = pd.read_csv(
            io.BytesIO(decoded)
        )

        page = html.Div([
            html.H5(
                "Text"
            ),
            #dcc.Graph(figure=px.histogram(df, x="Age", color="Exited")),
            dash_table.DataTable(
                data=df.to_dict(
                    orient="records"
                ),
                columns=[
                    {"name": i, "id": i} for i in df.columns
                ],
                page_size=5
            )
        ])
        return page


if __name__ == "__main__":
    app.run_server(debug=True)
