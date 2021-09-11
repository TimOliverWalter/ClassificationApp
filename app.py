import pandas as pd

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
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Dynamically rendered tab content"),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Sample data",
                    tab_id="sample-data"
                ),
                dbc.Tab(
                    label="New data",
                    tab_id="new-data"
                ),
            ],
            id="tabs",
            active_tab="sample-data",
        ),
        html.Div(
            id="tab-content"
        ),
    ]
)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    df = pd.read_csv("data/churn_modelling.csv")
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    df = df.replace(
        {"Geography": {"Germany": 0,
                       "France": 1,
                       "Spain": 2},
         "Gender": {"Female": 0,
                    "Male": 1}
         }
    )

    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_gbc)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted class", y="Actual class"),
        x=["P", "N"],
        y=["P", "N"]
    )
    fig.update_xaxes(side="top")

    accuracy = round(accuracy_score(y_test, y_pred_gbc), 4)
    f1 = round(f1_score(y_test, y_pred_gbc), 4)
    precision = round(precision_score(y_test, y_pred_gbc), 4)
    recall = round(recall_score(y_test, y_pred_gbc), 4)

    fig1 = go.Figure(
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

    """
    2nd option 
    
    table_header = [
        html.Thead(html.Tr([html.Th("metrics")]))
    ]
    row1 = html.Tr([html.Td("Arthur"), html.Td("Dent")])
    table_body = [html.Tbody([row1])]
    table = dbc.Table(table_header + table_body, bordered=True)
    """

    if active_tab is not None:
        if active_tab == "sample-data":
            return dbc.Row(
                html.Div(
                    [
                        html.H5("Exploratory Data Analysis (EDA)"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    daq.LEDDisplay(
                                        label="Records",
                                        value=len(df.index),
                                        backgroundColor="#FF5E5E"
                                    ),
                                    width=3
                                ),
                                dbc.Col(
                                    daq.LEDDisplay(
                                        label="Train",
                                        value=len(X_train.index),
                                        backgroundColor="#FF5E5E"
                                    ),
                                    width=3
                                ),
                                dbc.Col(
                                    daq.LEDDisplay(
                                        label="Test",
                                        value=len(X_test.index),
                                        backgroundColor="#FF5E5E"
                                    ),
                                    width=3
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        figure=fig
                                    ),
                                    width=6
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        figure=fig1
                                    ),
                                    width=6
                                )
                            ]
                        ),
                        dbc.Row(
                            dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in df.columns],
                                data=df.to_dict('records'),
                                page_size=5
                            )
                        )
                    ]
                )
            )
        elif active_tab == "new-data":
            return dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            figure=px.histogram(
                                df,
                                x="Age",
                                color="Exited"
                            )
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Graph(
                            figure=px.histogram(
                                df,
                                x="Age",
                                color="Exited"
                            )
                        ),
                        width=6
                    ),
                ]
            )
    return "No tab selected"


if __name__ == "__main__":
    app.run_server(debug=True)
