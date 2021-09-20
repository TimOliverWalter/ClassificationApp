import pandas as pd
import base64
import io

from pages import (
    overview,
    classifier_selection,
    example_data
)

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(external_stylesheets=["/assets/style.css"])

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
                        example_data.create_layout()
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
        dcc.Store(
            id="intermediate-value"
        )
    ]
)


@app.callback(
    Output("intermediate-value", "data"),
    Input("upload-page-one", "contents")
)
def parse_data(contents):
    df = pd.DataFrame()
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(
                io.BytesIO(decoded)
            )
            return df.to_dict(orient="records")
        except Exception as e:
            print(e)
    return df.to_dict(orient="records")


@app.callback(
    Output("output-page-one", "children"),
    Input("intermediate-value", "data")
)
def update_tab_one(dataset):
    # first check if the dataset is not empty
    if len(dataset) > 0:
        df = pd.DataFrame.from_dict(dataset)

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
    if len(dataset) > 0:
        df = pd.DataFrame.from_dict(dataset)
        classifier_tab = classifier_selection.create_layout(df=df)

        return classifier_tab
    return html.Div()


if __name__ == "__main__":
    app.run_server(debug=True)
