import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.df = pd.read_csv("data/churn_modelling.csv")
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

clf = RandomForestClassifier(random_state=10)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

result = pd.DataFrame(y_hat, columns=["Exited_Yhat"])
y_test = y_test.reset_index(drop=True)

result["Exited"] = y_test

print(result)
print(y_test)
print(X_test)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=None
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
