import dash_html_components as html
import dash_table


def create_layout(df, describe):
    return html.Div(
        [
            html.Div(
                [
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
                    )
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H6("Descriptive statistics"),
                            dash_table.DataTable(
                                data=describe.to_dict(
                                    orient="records"
                                ),
                                columns=[
                                    {"name": i, "id": i} for i in describe.columns
                                ]
                            ),

                        ], style={
                            "display": "inline-block"
                        }
                    ),
                ]
            )

        ]
    )
