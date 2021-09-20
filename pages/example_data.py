import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_html_components as html
import dash_table
import dash_core_components as dcc

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


def preparing_data(df):
    encod_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    dummy = pd.get_dummies(df[encod_cols], prefix=encod_cols, dtype=int)

    df = pd.merge(left=df, right=dummy, left_index=True, right_index=True)
    df = df.drop(encod_cols, axis=1)

    X = df.drop(["HeartDisease"], axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def create_figures(df):
    hist = create_hist(df=df)

    X_train, X_test, y_train, y_test = preparing_data(df=df)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    im = px.imshow(
        cm,
        labels=dict(
            x="Predicted class",
            y="Actual class"
        ),
        x=["P", "N"],
        y=["P", "N"]
    )
    im.update_xaxes(side="top")

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)

    table = go.Figure(
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

    return hist, im, table


def create_hist(df):
    return px.histogram(
        df,
        x="Age",
        color="Sex",
        pattern_shape="HeartDisease"
    )


def create_layout():
    path = "data/heart_failure.csv"
    df = pd.read_csv(path)

    hist, im, table = create_figures(df=df)

    layout = html.Div(
        [
            html.H5(
                "Text"
            ),
            dash_table.DataTable(
                data=df.to_dict(
                    orient="records"
                ),
                columns=[
                    {"name": i, "id": i} for i in df.columns
                ],
                page_size=5
            ),
            dcc.Graph(
                figure=hist
            ),
            dcc.Graph(
                figure=im
            ),
            dcc.Graph(
                figure=table
            )
        ]
    )
    return layout
