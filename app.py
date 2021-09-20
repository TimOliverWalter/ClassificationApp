import pandas as pd
import base64
import io

from pages import overview, classifier_selection

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
                ),
                dcc.Tab(
                    id="page-dree",
                    label="Classifier Selection",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    id="output-page-dree",
                                    children=[]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        dcc.Store(id="intermediate-value")
    ]
)


@app.callback(
    Output("intermediate-value", "data"),
    Input("upload-page-one", "contents")
)
def parse_data(contents):
    df =pd.DataFrame()
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        df = pd.read_csv(
            io.BytesIO(decoded)
        )
        return df.to_dict(orient="records")
    return df.to_dict(orient="records")


@app.callback(
    Output("output-page-one", "children"),
    Input("intermediate-value", "data")
)
def update_tab_one(dataset):
    print(len(dataset))
    if len(dataset) >0:
        df = pd.DataFrame.from_dict(dataset)
        X = df.drop(["Exited"], axis=1)
        y = df["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        missing_values_count = df.isnull().sum().to_frame()
        describe = df.describe(include="all")
        describe = describe.append(pd.Series(dtype="boolean", name="missing_values"))

        for index, row in missing_values_count.iterrows():
            print(index, row[0])
            if row[0] > 0:
                describe.loc["missing_values", index] = True
            else:
                describe.loc["missing_values", index] = False

        missing_values_count = missing_values_count.reset_index()
        missing_values_count.rename({0: "result", "index": "columns"}, axis=1)
        describe.insert(0, "", describe.index)

        overview_tab = overview.create_layout(df=df, describe=describe)

        return overview_tab
    return html.Div()


@app.callback(
    Output("output-page-dree", "children"),
    Input("intermediate-value", "data")
)
def update_tab_dree(dataset):
    df = pd.DataFrame.from_dict(dataset)
    classifier_tab = classifier_selection.create_layout(df=df)

    return classifier_tab


if __name__ == "__main__":
    app.run_server(debug=True)
