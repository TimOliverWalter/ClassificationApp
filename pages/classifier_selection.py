import dash_html_components as html
import dash_daq as daq
from sklearn.model_selection import train_test_split


def create_layout(df):
    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return html.Div(
        [
            html.Div(
                daq.LEDDisplay(
                    label="Records",
                    value=len(df.index),
                    backgroundColor="#FF5E5E"
                ),
                style={
                    "display": "inline-block"
                }
            ),
            html.Div(
                daq.LEDDisplay(
                    label="Train",
                    value=len(X_train.index),
                    backgroundColor="#FF5E5E"
                ),
                style={
                    "display": "inline-block",
                    "margin-left": "20px"
                }
            ),
            html.Div(
                daq.LEDDisplay(
                    label="Test",
                    value=len(X_test.index),
                    backgroundColor="#FF5E5E"
                ),
                style={
                    "display": "inline-block",
                    "margin-left": "20px"
                }
            ),
        ]
    )
