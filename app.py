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

data = pd.read_csv("data/heart_failure.csv")
fig = px.histogram(data, x="Age", color="Sex", pattern_shape="HeartDisease")

encod_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
dummy = pd.get_dummies(data[encod_cols], prefix=encod_cols, dtype=int)
data = pd.merge(
    left=data,
    right=dummy,
    left_index=True,
    right_index=True,
)
data = data.drop(encod_cols, axis=1)

X = data.drop(["HeartDisease"], axis=1)
y = data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
fig1 = px.imshow(
    cm,
    labels=dict(x="Predicted class", y="Actual class"),
    x=["P", "N"],
    y=["P", "N"]
)
fig1.update_xaxes(side="top")

accuracy = round(accuracy_score(y_test, y_pred), 4)
f1 = round(f1_score(y_test, y_pred), 4)
precision = round(precision_score(y_test, y_pred), 4)
recall = round(recall_score(y_test, y_pred), 4)

fig2 = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=["", "Metrics"]
            ),
            cells=dict(
                values=[
                    ["Accuracy", "F1", "Precision", "Recall"],
                    [accuracy, f1, precision, recall]
                ]
            )
        )
    ]
)

page2 = html.Div([
    html.H5(
        "Text"
    ),
    dash_table.DataTable(
        data=data.to_dict(
            orient="records"
        ),
        columns=[
            {"name": i, "id": i} for i in data.columns
        ],
        page_size=5
    ),
    dcc.Graph(figure=fig),
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2)
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
                                                        "Select the Data"
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
                                    },
                                    multiple=False
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
                    label="Heart Failure Data",
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
        X = df.drop(["Exited"], axis=1)
        y = df["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        page = html.Div([
            html.H5(
                "Text"
            ),
            # dcc.Graph(figure=px.histogram(df, x="Age", color="Exited")),
            dash_table.DataTable(
                data=df.to_dict(
                    orient="records"
                ),
                columns=[
                    {"name": i, "id": i} for i in df.columns
                ],
                page_size=5
            ),
            daq.LEDDisplay(
                label="Records",
                value=len(df.index),
                backgroundColor="#FF5E5E"
            ),
            daq.LEDDisplay(
                label="Train",
                value=len(X_train.index),
                backgroundColor="#FF5E5E"
            ),
            daq.LEDDisplay(
                label="Test",
                value=len(X_test.index),
                backgroundColor="#FF5E5E"
            )

        ])
        return page


if __name__ == "__main__":
    app.run_server(debug=True)
