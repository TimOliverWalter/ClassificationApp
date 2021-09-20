import dash_html_components as html
import dash_table
import dash_daq as daq


def create_layout(df, describe):
    return html.Div([
        html.Div([
            html.H6(
                "Dataset overview"
            ),
            dash_table.DataTable(
                data=df.to_dict(
                    orient="records"
                ),
                columns=[
                    {"name": i, "id": i} for i in df.columns
                ],
                page_size=5
            )]),

        # dcc.Graph(figure=px.histogram(df, x="Age", color="Exited")),

        html.Div([
            html.Div([
                html.H6("Stats"),
                dash_table.DataTable(
                    data=describe.to_dict(
                        orient="records"
                    ),
                    columns=[
                        {"name": i, "id": i} for i in describe.columns
                    ], style_table={"minWidth": "100%"}
                ),

            ], style={"display": "inline-block"}),
        ]),
        html.Div([
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

    ])
