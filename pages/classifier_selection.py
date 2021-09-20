import dash_html_components as html
import dash_daq as daq

def create_layout(df):
    return html.Div([
            html.Div(
                daq.LEDDisplay(
                    label="Records",
                    value=len(df.index),
                    backgroundColor="#FF5E5E"
                ), style={"display": "inline-block"}),
            html.Div(
                daq.LEDDisplay(
                    label="Records",
                    value=len(df.index),
                    backgroundColor="#FF5E5E"
                ), style={"display": "inline-block"}),
            html.Div(
                daq.LEDDisplay(
                    label="Records",
                    value=len(df.index),
                    backgroundColor="#FF5E5E"
                ), style={"display": "inline-block", "margin-left": "10px"}),
        ])