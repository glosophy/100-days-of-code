# This app tells you if Chick-Fil-A is open

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Img(id='image',
             src="https://cdn.clipart.email/49ee1e3c8f46800efe4d7e2f01658681_transparent-chick-fil-a-clipart_317-317.png",
             style={'position': 'relative',
                    'margin': 'auto',
                    'display': 'block',
                    'margin-top': '70px'
                    }),

    html.Div([
        html.Button('Is Chick-Fil-A open?', id='btn-nclicks-1', n_clicks=0,
                    style={
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-top': '50px',
                        'margin-left': '38%'
                    }),

        html.H1(id='container-button-timestamp',
                style={
                    'position': 'relative',
                    'font-weight': 'bold',
                    'margin-left': '44%',
                    'margin-top': '50px'
                }),

        dcc.Markdown('''
                Brought to you by GSS ♥.
                ''',

                     style={'position': 'relative',
                            'textAlign': 'center',
                            'margin-top': '300px'
                            })

    ])

], style={'margin': 'auto',
          'display': 'block',
          'width': '50%'
          })


@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks')
              )
def displayClick(btn1):
    global result
    now = datetime.datetime.now()
    msg = now.strftime("%A")

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        if msg == 'Sunday':
            result = 'NO'
        else:
            result = 'YES!'

    return html.Div(result)


if __name__ == '__main__':
    app.run_server(debug=True)
