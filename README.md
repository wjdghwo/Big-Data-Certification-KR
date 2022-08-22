# Big-Data-Certification-KR

빅데이터 분석기사 실기 시험을 준비하며 연습한 코드들입니다.

주로 다음의 kaggle data를 사용하여 학습했습니다.
https://www.kaggle.com/agileteam/bigdatacertificationkr














import numpy as np
import pandas as pd
import random
from random import randint

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


row = ['취급건수', '취급액', '약정금리', '기준금리', '조달원가', 
       'Duration', '업무원가', '공통원가', '채널별가산금리', '모집수수료', 
       '신용원가', '목표이익률', '조정금리', '우대금리', '프로모션금리', 
       '상품운영금리', '조달조정금리', '적용이익율']

col = pd.date_range('2022-07-01', '2022-07-31')

channel = ['전체', '제휴연계외', '제휴연계', 
           '상담사', 'direct', 
           '카카오', '엘포인트', '토스', '핀다', '카카오페이', 
           '핀크', '시럽', '알다', '신한카드', '케이뱅크', 
           '뱅크샐러드', '하나카드', '페이코', '키움증권', 'NICE평가정보']

data = pd.DataFrame()

for i in range(len(channel)):
    data_temp = pd.DataFrame({'취급건수':np.random.binomial(400, 0.5, size=len(col)),
                        '취급액':np.random.binomial(10000, 0.4, size=len(col)),
                        '약정금리':np.random.normal(12, 1, size=len(col)),
                        '기준금리':np.random.normal(14, 1, size=len(col)),
                        '조달원가':np.random.normal(4, 0.5, size=len(col)),
                        'Duration':np.random.normal(43, 2, size=len(col)),
                        '업무원가':np.random.normal(2.5, 0.3, size=len(col)),
                        '공통원가':np.random.normal(1.4, 0.1, size=len(col)),
                        '채널별가산금리':abs(np.random.normal(0.1, 0.1, size=len(col))),
                        '모집수수료':abs(np.random.normal(1, 0.1, len(col))),
                        '신용원가':np.random.normal(5, 0.5, len(col)),
                        '목표이익률':np.random.normal(3.2, 0.1, len(col)),
                        '조정금리':abs(np.random.normal(2.3, 0.2, len(col)))*(-1),
                        '우대금리':abs(np.random.normal(0, 0.05, len(col)))*(-1),
                        '프로모션금리':abs(np.random.normal(1.8, 0.3, len(col)))*(-1),
                        '상품운영금리':abs(np.random.normal(0.5, 0.2, len(col)))*(-1),
                        '조달조정금리':abs(np.random.normal(0.01, 0.01, len(col)))*(-1),
                        '적용이익율':abs(np.random.normal(0.8, 0.1, len(col)))}).T
    data_temp.columns=col
    data_temp['channel'] = channel[i]
    data=pd.concat([data,data_temp],axis=0)

df = data1.reset_index().melt(id_vars=['index','channel'], var_name = ['date']).rename(columns = {'index':'type'})


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

markdown_text = '''
# 개인신용대출 금리원가별 Monitoring
### 작성기준

* 대상기준 : 개인신용대출 신규상품 당일 실행 및 당일 출금 고객 (실행일 이후 수시출입금, 당일취소 제외)
* 집계기준 : 원가별 출금액 가준평균금리
* 고객별 예상 마진율 : 약정금리 - sum(조달원가, 업무원가, 신용원가)
'''

dcc.Markdown(children=markdown_text)

app.layout = html.Div([
    html.Div([
        dcc.Markdown(children=markdown_text)
    ]),

    
    html.Div([

        html.Div([
            dcc.Dropdown(
                df['channel'].unique(),
                'Fertility rate, total (births per woman)',
                id='crossfilter-xaxis-column',
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-xaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df['channel'].unique(),
                'Life expectancy at birth, total (dates)',
                id='crossfilter-yaxis-column'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),
    
    # 우측상단 그래프
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': '적용이익율'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    
    # 우측하단 그래프
    html.Div(dcc.Slider(
        df['channel'].min(),
        df['date'].max(),
        step=None,
        id='crossfilter-date--slider',
        value=df['date'].max(),
        marks={str(date): str(date) for date in df['date'].unique()}
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-date--slider', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 date_value):
    dff = df[df['date'] == date_value]

    fig = px.scatter(x=dff[dff['channel'] == xaxis_column_name]['value'],
            y=dff[dff['channel'] == yaxis_column_name]['value'],
            hover_name=dff[dff['channel'] == yaxis_column_name]['type']
            )

    fig.update_traces(customdata=dff[dff['channel'] == yaxis_column_name]['type'])

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='date', y='value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'))
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['type'] == country_name]
    dff = dff[dff['channel'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@app.callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['type'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['channel'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)

if __name__ == '__main__':
    app.run_server(host='localhost',port=8007)














































#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import pandas as pd
import random
from random import randint


# In[121]:


row = ['취급건수', '취급액', '약정금리', '기준금리', '조달원가', 
       'Duration', '업무원가', '공통원가', '채널별가산금리', '모집수수료', 
       '신용원가', '목표이익률', '조정금리', '우대금리', '프로모션금리', 
       '상품운영금리', '조달조정금리', '적용이익율']

col = pd.date_range('2022-07-01', '2022-07-31')


# In[122]:


channel = ['전체', '제휴연계외', '제휴연계', 
           '상담사', 'direct', 
           '카카오', '엘포인트', '토스', '핀다', '카카오페이', 
           '핀크', '시럽', '알다', '신한카드', '케이뱅크', 
           '뱅크샐러드', '하나카드', '페이코', '키움증권', 'NICE평가정보']


# In[123]:


data = pd.DataFrame()

for i in range(len(channel)):
    data_temp = pd.DataFrame({'취급건수':np.random.binomial(400, 0.5, size=len(col)),
                        '취급액':np.random.binomial(10000, 0.4, size=len(col)),
                        '약정금리':np.random.normal(12, 1, size=len(col)),
                        '기준금리':np.random.normal(14, 1, size=len(col)),
                        '조달원가':np.random.normal(4, 0.5, size=len(col)),
                        'Duration':np.random.normal(43, 2, size=len(col)),
                        '업무원가':np.random.normal(2.5, 0.3, size=len(col)),
                        '공통원가':np.random.normal(1.4, 0.1, size=len(col)),
                        '채널별가산금리':abs(np.random.normal(0.1, 0.1, size=len(col))),
                        '모집수수료':abs(np.random.normal(1, 0.1, len(col))),
                        '신용원가':np.random.normal(5, 0.5, len(col)),
                        '목표이익률':np.random.normal(3.2, 0.1, len(col)),
                        '조정금리':abs(np.random.normal(2.3, 0.2, len(col)))*(-1),
                        '우대금리':abs(np.random.normal(0, 0.05, len(col)))*(-1),
                        '프로모션금리':abs(np.random.normal(1.8, 0.3, len(col)))*(-1),
                        '상품운영금리':abs(np.random.normal(0.5, 0.2, len(col)))*(-1),
                        '조달조정금리':abs(np.random.normal(0.01, 0.01, len(col)))*(-1),
                        '적용이익율':abs(np.random.normal(0.8, 0.1, len(col)))}).T
    data_temp.columns=col
    data_temp['channel'] = channel[i]
    data=pd.concat([data,data_temp],axis=0)


# In[124]:


data = data.reset_index().rename(columns={'index': 'type'})


# In[128]:


data


# In[125]:


df = pd.DataFrame()
for i in range(len(col)):
    temp = data.iloc[:,[0,i+1,-1]]
    temp['date'] = temp.columns[1]
    temp.columns = ['type', 'value', 'channel', 'date']
    df = pd.concat([df,temp],axis=0)


# In[126]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)





app = Dash(__name__)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("개인신용대출 금리원가별 Monitoring", style={'text-align': 'center'}),

    dcc.Dropdown(id="my-dropdown",
                 options=[
                     {"label": "취급건수", "value": '취급건수'},
                     {"label": "취급액", "value": "취급액"},
                     {"label": "약정금리", "value": '약정금리'},
                     {"label": "기준금리", "value": '기준금리'}],
                 multi=False,
                 value='취급건수',
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my-graph', figure={})

])


# In[ ]:





# In[103]:


a=pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')


# In[107]:


a[a['Country Name'] =='Japan']


# In[119]:


df


# In[101]:


df


# $\sum$

# In[ ]:


from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

markdown_text = '''
# 개인신용대출 금리원가별 Monitoring
### 작성기준

* 대상기준 : 개인신용대출 신규상품 당일 실행 및 당일 출금 고객 (실행일 이후 수시출입금, 당일취소 제외)
* 집계기준 : 원가별 출금액 가준평균금리
* 고객별 예상 마진율 : 약정금리 - sum(조달원가, 업무원가, 신용원가)
'''

dcc.Markdown(children=markdown_text)

app.layout = html.Div([
    html.Div([
        dcc.Markdown(children=markdown_text)
    ]),

    
    html.Div([

        html.Div([
            dcc.Dropdown(
                df['channel'].unique(),
                'Fertility rate, total (births per woman)',
                id='crossfilter-xaxis-column',
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-xaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df['channel'].unique(),
                'Life expectancy at birth, total (dates)',
                id='crossfilter-yaxis-column'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),
    
    # 우측상단 그래프
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': '적용이익율'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    
    # 우측하단 그래프
    html.Div(dcc.Slider(
        df['channel'].min(),
        df['date'].max(),
        step=None,
        id='crossfilter-date--slider',
        value=df['date'].max(),
        marks={str(date): str(date) for date in df['date'].unique()}
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-date--slider', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 date_value):
    dff = df[df['date'] == date_value]

    fig = px.scatter(x=dff[dff['channel'] == xaxis_column_name]['value'],
            y=dff[dff['channel'] == yaxis_column_name]['value'],
            hover_name=dff[dff['channel'] == yaxis_column_name]['type']
            )

    fig.update_traces(customdata=dff[dff['channel'] == yaxis_column_name]['type'])

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='date', y='value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'))
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['type'] == country_name]
    dff = dff[dff['channel'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@app.callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['type'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['channel'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)

if __name__ == '__main__':
    app.run_server(host='localhost',port=8007)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[180]:


import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)





app = Dash(__name__)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("개인신용대출 금리원가별 Monitoring", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2015,
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["Year"] == option_slctd]
    dff = dff[dff["Affected by"] == "Varroa_mites"]

    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )

#     Plotly Graph Objects (GO)
    fig = go.Figure(
        data=[go.Choropleth(
            locationmode='USA-states',
            locations=dff['state_code'],
            z=dff["Pct of Colonies Impacted"].astype(float),
            colorscale='Reds',
        )]
    )
    
    fig.update_layout(
        title_text="Bees Affected by Mites in the USA",
        title_xanchor="center",
        title_font=dict(size=24),
        title_x=0.5,
        geo=dict(scope='usa'),
    )

    return container, fig 


# ------------------------------------------------------------------------------
# port 설정 http://localhost:8005/

if __name__ == '__main__':
    app.run_server(host='localhost',port=8005)


# In[ ]:





# In[184]:


import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

df = pd.DataFrame(
    {
        "Fruit": [
            "Apples",
            "Oranges",
            "Bananas",
            "Apples",
            "Oranges",
            "Bananas",
        ],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"],
    }
)

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(
    children=[
        html.H1(children='Hello Dash'),
        html.Div(
            children='''
        Dash: A web application framework for your data.
    '''
        ),
        dcc.Graph(id='example-graph', figure=fig),
    ]
)

if __name__ == '__main__':
    app.run_server(host='localhost',port=8005)


# In[ ]:


import dash
from dash import dcc
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            children=[
                html.Label('Dropdown'),
                dcc.Dropdown(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'},
                    ],
                    value='MTL',
                ),
                html.Br(),
                html.Label('Multi-Select Dropdown'),
                dcc.Dropdown(
                    options=[z
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'},
                    ],
                    value=['MTL', 'SF'],
                    multi=True,
                ),
                html.Br(),
                html.Label('Radio Items'),
                dcc.RadioItems(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'},
                    ],
                    value='MTL',
                ),
            ],
            style={'padding': 10, 'flex': 1},
        ),
        html.Div(
            children=[
                html.Label('Checkboxes'),
                dcc.Checklist(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'},
                    ],
                    value=['MTL', 'SF'],
                ),
                html.Br(),
                html.Label('Text Input'),
                dcc.Input(value='MTL', type='text'),
                html.Br(),
                html.Label('Slider'),
                dcc.Slider(
                    min=0,
                    max=9,
                    marks={
                        i: 'Label {}'.format(i) if i == 1 else str(i)
                        for i in range(1, 6)
                    },
                    value=5,
                ),
            ],
            style={'padding': 10, 'flex': 1},
        ),
    ],
    style={'display': 'flex', 'flex-direction': 'row'},
)

if __name__ == '__main__':
    app.run_server(host='localhost',port=8005)
    


# In[ ]:





# In[ ]:





# In[161]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


@app.callback(
    Output('intermediate-value', 'data'),
    Input('dropdown', 'value'))
def clean_data(value):
     cleaned_df = slow_processing_step(value)

     # a few filter steps that compute the data
     # as it's needed in the future callbacks
     df_1 = cleaned_df[cleaned_df['fruit'] == 'apples']
     df_2 = cleaned_df[cleaned_df['fruit'] == 'oranges']
     df_3 = cleaned_df[cleaned_df['fruit'] == 'figs']

     datasets = {
         'df_1': df_1.to_json(orient='split', date_format='iso'),
         'df_2': df_2.to_json(orient='split', date_format='iso'),
         'df_3': df_3.to_json(orient='split', date_format='iso'),
     }

     return json.dumps(datasets)

@app.callback(
    Output('graph1', 'figure'),
    Input('intermediate-value', 'data'))
def update_graph_1(jsonified_cleaned_data):
    datasets = json.loads(jsonified_cleaned_data)
    dff = pd.read_json(datasets['df_1'], orient='split')
    figure = create_figure_1(dff)
    return figure

@app.callback(
    Output('graph2', 'figure'),
    Input('intermediate-value', 'data'))
def update_graph_2(jsonified_cleaned_data):
    datasets = json.loads(jsonified_cleaned_data)
    dff = pd.read_json(datasets['df_2'], orient='split')
    figure = create_figure_2(dff)
    return figure

@app.callback(
    Output('graph3', 'figure'),
    Input('intermediate-value', 'data'))
def update_graph_3(jsonified_cleaned_data):
    datasets = json.loads(jsonified_cleaned_data)
    dff = pd.read_json(datasets['df_3'], orient='split')
    figure = create_figure_3(dff)
    return figure


# In[ ]:





# In[ ]:






































aiohttp @ file:///C:/ci/aiohttp_1646806572557/work
aiosignal @ file:///tmp/build/80754af9/aiosignal_1637843061372/work
alabaster @ file:///home/ktietz/src/ci/alabaster_1611921544520/work
anaconda-client @ file:///C:/ci/anaconda-client_1635342725944/work
anaconda-navigator==2.1.4
anaconda-project @ file:///tmp/build/80754af9/anaconda-project_1637161053845/work
anyio @ file:///C:/ci/anyio_1644481921011/work/dist
appdirs==1.4.4
argon2-cffi @ file:///opt/conda/conda-bld/argon2-cffi_1645000214183/work
argon2-cffi-bindings @ file:///C:/ci/argon2-cffi-bindings_1644551690056/work
arrow @ file:///opt/conda/conda-bld/arrow_1649166651673/work
astroid @ file:///C:/ci/astroid_1628063282661/work
astropy @ file:///C:/ci/astropy_1650634291321/work
asttokens @ file:///opt/conda/conda-bld/asttokens_1646925590279/work
async-timeout @ file:///tmp/build/80754af9/async-timeout_1637851218186/work
atomicwrites==1.4.0
attrs @ file:///opt/conda/conda-bld/attrs_1642510447205/work
Automat @ file:///tmp/build/80754af9/automat_1600298431173/work
autopep8 @ file:///opt/conda/conda-bld/autopep8_1639166893812/work
Babel @ file:///tmp/build/80754af9/babel_1620871417480/work
backcall @ file:///home/ktietz/src/ci/backcall_1611930011877/work
backports.functools-lru-cache @ file:///tmp/build/80754af9/backports.functools_lru_cache_1618170165463/work
backports.tempfile @ file:///home/linux1/recipes/ci/backports.tempfile_1610991236607/work
backports.weakref==1.0.post1
bcrypt @ file:///C:/ci/bcrypt_1607022693089/work
beautifulsoup4 @ file:///C:/ci/beautifulsoup4_1650293025093/work
binaryornot @ file:///tmp/build/80754af9/binaryornot_1617751525010/work
bitarray @ file:///C:/ci/bitarray_1648739663053/work
bkcharts==0.2
black==19.10b0
bleach @ file:///opt/conda/conda-bld/bleach_1641577558959/work
bokeh @ file:///C:/ci/bokeh_1638362966927/work
boto3 @ file:///opt/conda/conda-bld/boto3_1649078879353/work
botocore @ file:///opt/conda/conda-bld/botocore_1649076662316/work
Bottleneck @ file:///C:/ci/bottleneck_1648010904582/work
Brotli==1.0.9
brotlipy==0.7.0
cachetools @ file:///tmp/build/80754af9/cachetools_1619597386817/work
certifi==2021.10.8
cffi @ file:///C:/ci_310/cffi_1642682485096/work
chardet @ file:///C:/ci/chardet_1607706937985/work
charset-normalizer @ file:///tmp/build/80754af9/charset-normalizer_1630003229654/work
click @ file:///C:/ci/click_1646038595831/work
cloudpickle @ file:///tmp/build/80754af9/cloudpickle_1632508026186/work
clyent==1.2.2
colorama @ file:///tmp/build/80754af9/colorama_1607707115595/work
colorcet @ file:///tmp/build/80754af9/colorcet_1611168489822/work
comtypes==1.1.10
conda==4.12.0
conda-build==3.21.8
conda-content-trust @ file:///tmp/build/80754af9/conda-content-trust_1617045594566/work
conda-pack @ file:///tmp/build/80754af9/conda-pack_1611163042455/work
conda-package-handling @ file:///C:/ci/conda-package-handling_1649106011304/work
conda-repo-cli @ file:///tmp/build/80754af9/conda-repo-cli_1620168426516/work
conda-token @ file:///tmp/build/80754af9/conda-token_1620076980546/work
conda-verify==3.4.2
constantly==15.1.0
cookiecutter @ file:///opt/conda/conda-bld/cookiecutter_1649151442564/work
cryptography @ file:///C:/ci/cryptography_1633520531101/work
cssselect==1.1.0
cycler @ file:///tmp/build/80754af9/cycler_1637851556182/work
Cython @ file:///C:/ci/cython_1647850559892/work
cytoolz==0.11.0
daal==2021.4.0
daal4py==2021.5.0
dash==2.6.1
dash-bootstrap-components==1.2.1
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
dask @ file:///opt/conda/conda-bld/dask-core_1647268715755/work
datashader @ file:///tmp/build/80754af9/datashader_1623782308369/work
datashape==0.5.4
debugpy @ file:///C:/ci/debugpy_1637091961445/work
decorator @ file:///opt/conda/conda-bld/decorator_1643638310831/work
defusedxml @ file:///tmp/build/80754af9/defusedxml_1615228127516/work
diff-match-patch @ file:///Users/ktietz/demo/mc3/conda-bld/diff-match-patch_1630511840874/work
distributed @ file:///opt/conda/conda-bld/distributed_1647271944416/work
docutils @ file:///C:/ci/docutils_1620828264669/work
entrypoints @ file:///C:/ci/entrypoints_1649926621128/work
et-xmlfile==1.1.0
executing @ file:///opt/conda/conda-bld/executing_1646925071911/work
fastjsonschema @ file:///tmp/build/80754af9/python-fastjsonschema_1620414857593/work/dist
filelock @ file:///opt/conda/conda-bld/filelock_1647002191454/work
flake8 @ file:///tmp/build/80754af9/flake8_1620776156532/work
Flask @ file:///home/ktietz/src/ci/flask_1611932660458/work
Flask-Compress==1.12
fonttools==4.25.0
frozenlist @ file:///C:/ci/frozenlist_1637767271796/work
fsspec @ file:///opt/conda/conda-bld/fsspec_1647268051896/work
future @ file:///C:/ci/future_1607568713721/work
gensim @ file:///C:/ci/gensim_1646825438310/work
glob2 @ file:///home/linux1/recipes/ci/glob2_1610991677669/work
google-api-core @ file:///C:/ci/google-api-core-split_1613980333946/work
google-auth @ file:///tmp/build/80754af9/google-auth_1626320605116/work
google-cloud-core @ file:///tmp/build/80754af9/google-cloud-core_1625077425256/work
google-cloud-storage @ file:///tmp/build/80754af9/google-cloud-storage_1601307969662/work
google-crc32c @ file:///C:/ci/google-crc32c_1613234249694/work
google-resumable-media @ file:///tmp/build/80754af9/google-resumable-media_1624367812531/work
googleapis-common-protos @ file:///C:/ci/googleapis-common-protos-feedstock_1617957814607/work
greenlet @ file:///C:/ci/greenlet_1628888275363/work
grpcio @ file:///C:/ci/grpcio_1637590978642/work
h5py @ file:///C:/ci/h5py_1637120894255/work
HeapDict @ file:///Users/ktietz/demo/mc3/conda-bld/heapdict_1630598515714/work
holoviews @ file:///opt/conda/conda-bld/holoviews_1645454331194/work
httplib2==0.20.4
hvplot @ file:///tmp/build/80754af9/hvplot_1627305124151/work
hyperlink @ file:///tmp/build/80754af9/hyperlink_1610130746837/work
idna @ file:///tmp/build/80754af9/idna_1637925883363/work
imagecodecs @ file:///C:/ci/imagecodecs_1635511087451/work
imageio @ file:///tmp/build/80754af9/imageio_1617700267927/work
imagesize @ file:///tmp/build/80754af9/imagesize_1637939814114/work
importlib-metadata @ file:///C:/ci/importlib-metadata_1648562621412/work
incremental @ file:///tmp/build/80754af9/incremental_1636629750599/work
inflection==0.5.1
iniconfig @ file:///home/linux1/recipes/ci/iniconfig_1610983019677/work
intake @ file:///opt/conda/conda-bld/intake_1647436631684/work
intervaltree @ file:///Users/ktietz/demo/mc3/conda-bld/intervaltree_1630511889664/work
ipykernel @ file:///C:/ci/ipykernel_1646982785443/work/dist/ipykernel-6.9.1-py3-none-any.whl
ipython @ file:///C:/ci/ipython_1648817223581/work
ipython-genutils @ file:///tmp/build/80754af9/ipython_genutils_1606773439826/work
ipywidgets @ file:///tmp/build/80754af9/ipywidgets_1634143127070/work
isort @ file:///tmp/build/80754af9/isort_1628603791788/work
itemadapter @ file:///tmp/build/80754af9/itemadapter_1626442940632/work
itemloaders @ file:///opt/conda/conda-bld/itemloaders_1646805235997/work
itsdangerous @ file:///tmp/build/80754af9/itsdangerous_1621432558163/work
jdcal @ file:///Users/ktietz/demo/mc3/conda-bld/jdcal_1630584345063/work
jedi @ file:///C:/ci/jedi_1644315428289/work
Jinja2 @ file:///tmp/build/80754af9/jinja2_1612213139570/work
jinja2-time @ file:///opt/conda/conda-bld/jinja2-time_1649251842261/work
jmespath @ file:///Users/ktietz/demo/mc3/conda-bld/jmespath_1630583964805/work
joblib @ file:///tmp/build/80754af9/joblib_1635411271373/work
json5 @ file:///tmp/build/80754af9/json5_1624432770122/work
jsonschema @ file:///C:/ci/jsonschema_1650008058050/work
jupyter @ file:///C:/ci/jupyter_1607685287094/work
jupyter-client @ file:///tmp/build/80754af9/jupyter_client_1616770841739/work
jupyter-console @ file:///tmp/build/80754af9/jupyter_console_1616615302928/work
jupyter-core @ file:///C:/ci/jupyter_core_1646994619043/work
jupyter-server @ file:///opt/conda/conda-bld/jupyter_server_1644494914632/work
jupyterlab @ file:///opt/conda/conda-bld/jupyterlab_1647445413472/work
jupyterlab-pygments @ file:///tmp/build/80754af9/jupyterlab_pygments_1601490720602/work
jupyterlab-server @ file:///opt/conda/conda-bld/jupyterlab_server_1644500396812/work
jupyterlab-widgets @ file:///tmp/build/80754af9/jupyterlab_widgets_1609884341231/work
keyring @ file:///C:/ci/keyring_1638531673471/work
kiwisolver @ file:///C:/ci/kiwisolver_1644962577370/work
lazy-object-proxy @ file:///C:/ci/lazy-object-proxy_1616529288960/work
libarchive-c @ file:///tmp/build/80754af9/python-libarchive-c_1617780486945/work
llvmlite==0.38.0
locket @ file:///C:/ci/locket_1647006279389/work
lxml @ file:///C:/ci/lxml_1646642862366/work
Markdown @ file:///C:/ci/markdown_1614364082838/work
MarkupSafe @ file:///C:/ci/markupsafe_1621528502553/work
matplotlib @ file:///C:/ci/matplotlib-suite_1647423638658/work
matplotlib-inline @ file:///tmp/build/80754af9/matplotlib-inline_1628242447089/work
mccabe==0.6.1
menuinst @ file:///C:/ci/menuinst_1631733438520/work
mistune @ file:///C:/ci/mistune_1607359457024/work
mkl-fft==1.3.1
mkl-random @ file:///C:/ci/mkl_random_1626186184308/work
mkl-service==2.4.0
mock @ file:///tmp/build/80754af9/mock_1607622725907/work
mpmath==1.2.1
msgpack @ file:///C:/ci/msgpack-python_1612287350784/work
multidict @ file:///C:/ci/multidict_1607349747897/work
multipledispatch @ file:///C:/ci/multipledispatch_1607574329826/work
munkres==1.1.4
mypy-extensions==0.4.3
navigator-updater==0.2.1
nbclassic @ file:///opt/conda/conda-bld/nbclassic_1644943264176/work
nbclient @ file:///C:/ci/nbclient_1650290387259/work
nbconvert @ file:///C:/ci/nbconvert_1649741016669/work
nbformat @ file:///C:/ci/nbformat_1649845125000/work
nest-asyncio @ file:///C:/ci/nest-asyncio_1649829929390/work
networkx @ file:///opt/conda/conda-bld/networkx_1647437648384/work
nltk @ file:///opt/conda/conda-bld/nltk_1645628263994/work
nose @ file:///opt/conda/conda-bld/nose_1642704612149/work
notebook @ file:///C:/ci/notebook_1645002729033/work
numba @ file:///C:/ci/numba_1650394399948/work
numexpr @ file:///C:/ci/numexpr_1640704337920/work
numpy==1.23.2
numpydoc @ file:///opt/conda/conda-bld/numpydoc_1643788541039/work
olefile @ file:///Users/ktietz/demo/mc3/conda-bld/olefile_1629805411829/work
openpyxl @ file:///tmp/build/80754af9/openpyxl_1632777717936/work
packaging @ file:///tmp/build/80754af9/packaging_1637314298585/work
pandas==1.4.3
pandocfilters @ file:///opt/conda/conda-bld/pandocfilters_1643405455980/work
panel @ file:///C:/ci/panel_1650623703033/work
param @ file:///tmp/build/80754af9/param_1636647414893/work
paramiko @ file:///opt/conda/conda-bld/paramiko_1640109032755/work
parsel @ file:///C:/ci/parsel_1646740216444/work
parso @ file:///opt/conda/conda-bld/parso_1641458642106/work
partd @ file:///opt/conda/conda-bld/partd_1647245470509/work
pathspec==0.7.0
patsy==0.5.2
pep8==1.7.1
pexpect @ file:///tmp/build/80754af9/pexpect_1605563209008/work
pickleshare @ file:///tmp/build/80754af9/pickleshare_1606932040724/work
Pillow==9.0.1
pkginfo @ file:///tmp/build/80754af9/pkginfo_1643162084911/work
plotly==5.9.0
pluggy @ file:///C:/ci/pluggy_1648024580010/work
poyo @ file:///tmp/build/80754af9/poyo_1617751526755/work
prometheus-client @ file:///opt/conda/conda-bld/prometheus_client_1643788673601/work
prompt-toolkit @ file:///tmp/build/80754af9/prompt-toolkit_1633440160888/work
Protego @ file:///tmp/build/80754af9/protego_1598657180827/work
protobuf==3.19.1
psutil @ file:///C:/ci/psutil_1612298199233/work
ptyprocess @ file:///tmp/build/80754af9/ptyprocess_1609355006118/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl
pure-eval @ file:///opt/conda/conda-bld/pure_eval_1646925070566/work
py @ file:///opt/conda/conda-bld/py_1644396412707/work
pyasn1 @ file:///Users/ktietz/demo/mc3/conda-bld/pyasn1_1629708007385/work
pyasn1-modules==0.2.8
pycodestyle @ file:///tmp/build/80754af9/pycodestyle_1615748559966/work
pycosat==0.6.3
pycparser @ file:///tmp/build/80754af9/pycparser_1636541352034/work
pyct @ file:///C:/ci/pyct_1613411728548/work
pycurl==7.44.1
PyDispatcher==2.0.5
pydocstyle @ file:///tmp/build/80754af9/pydocstyle_1621600989141/work
pyerfa @ file:///C:/ci/pyerfa_1621560974055/work
pyflakes @ file:///tmp/build/80754af9/pyflakes_1617200973297/work
Pygments @ file:///opt/conda/conda-bld/pygments_1644249106324/work
PyHamcrest @ file:///tmp/build/80754af9/pyhamcrest_1615748656804/work
PyJWT @ file:///C:/ci/pyjwt_1619682721924/work
pylint @ file:///C:/ci/pylint_1627536884966/work
pyls-spyder==0.4.0
PyNaCl @ file:///C:/ci/pynacl_1607612759007/work
pyodbc @ file:///C:/ci/pyodbc_1647426110990/work
pyOpenSSL @ file:///tmp/build/80754af9/pyopenssl_1635333100036/work
pyparsing @ file:///tmp/build/80754af9/pyparsing_1635766073266/work
pyreadline==2.1
pyrsistent @ file:///C:/ci/pyrsistent_1636093225342/work
PySocks @ file:///C:/ci/pysocks_1605307512533/work
pytest==7.1.1
python-dateutil @ file:///tmp/build/80754af9/python-dateutil_1626374649649/work
python-lsp-black @ file:///tmp/build/80754af9/python-lsp-black_1634232156041/work
python-lsp-jsonrpc==1.0.0
python-lsp-server==1.2.4
python-slugify @ file:///tmp/build/80754af9/python-slugify_1620405669636/work
python-snappy @ file:///C:/ci/python-snappy_1610133405910/work
pytz==2021.3
pyviz-comms @ file:///tmp/build/80754af9/pyviz_comms_1623747165329/work
PyWavelets @ file:///C:/ci/pywavelets_1648728084106/work
pywin32==302
pywin32-ctypes @ file:///C:/ci/pywin32-ctypes_1607553594546/work
pywinpty @ file:///C:/ci_310/pywinpty_1644230983541/work/target/wheels/pywinpty-2.0.2-cp39-none-win_amd64.whl
PyYAML==6.0
pyzmq @ file:///C:/ci/pyzmq_1638435148211/work
QDarkStyle @ file:///tmp/build/80754af9/qdarkstyle_1617386714626/work
qstylizer @ file:///tmp/build/80754af9/qstylizer_1617713584600/work/dist/qstylizer-0.1.10-py2.py3-none-any.whl
QtAwesome @ file:///tmp/build/80754af9/qtawesome_1637160816833/work
qtconsole @ file:///opt/conda/conda-bld/qtconsole_1649078897110/work
QtPy @ file:///opt/conda/conda-bld/qtpy_1649073884068/work
queuelib==1.5.0
regex @ file:///C:/ci/regex_1648447888413/work
requests @ file:///opt/conda/conda-bld/requests_1641824580448/work
requests-file @ file:///Users/ktietz/demo/mc3/conda-bld/requests-file_1629455781986/work
rope @ file:///opt/conda/conda-bld/rope_1643788605236/work
rsa @ file:///tmp/build/80754af9/rsa_1614366226499/work
Rtree @ file:///C:/ci/rtree_1618421015405/work
ruamel-yaml-conda @ file:///C:/ci/ruamel_yaml_1616016898638/work
s3transfer @ file:///tmp/build/80754af9/s3transfer_1626435152308/work
scikit-image @ file:///C:/ci/scikit-image_1648214340990/work
scikit-learn @ file:///C:/ci/scikit-learn_1642617276183/work
scikit-learn-intelex==2021.20220215.102710
scipy @ file:///C:/ci/scipy_1641555170412/work
Scrapy @ file:///C:/ci/scrapy_1646837986255/work
seaborn @ file:///tmp/build/80754af9/seaborn_1629307859561/work
Send2Trash @ file:///tmp/build/80754af9/send2trash_1632406701022/work
service-identity @ file:///Users/ktietz/demo/mc3/conda-bld/service_identity_1629460757137/work
sip==4.19.13
six @ file:///tmp/build/80754af9/six_1644875935023/work
smart-open @ file:///tmp/build/80754af9/smart_open_1623928409369/work
sniffio @ file:///C:/ci/sniffio_1614030527509/work
snowballstemmer @ file:///tmp/build/80754af9/snowballstemmer_1637937080595/work
sortedcollections @ file:///tmp/build/80754af9/sortedcollections_1611172717284/work
sortedcontainers @ file:///tmp/build/80754af9/sortedcontainers_1623949099177/work
soupsieve @ file:///tmp/build/80754af9/soupsieve_1636706018808/work
Sphinx @ file:///opt/conda/conda-bld/sphinx_1643644169832/work
sphinxcontrib-applehelp @ file:///home/ktietz/src/ci/sphinxcontrib-applehelp_1611920841464/work
sphinxcontrib-devhelp @ file:///home/ktietz/src/ci/sphinxcontrib-devhelp_1611920923094/work
sphinxcontrib-htmlhelp @ file:///tmp/build/80754af9/sphinxcontrib-htmlhelp_1623945626792/work
sphinxcontrib-jsmath @ file:///home/ktietz/src/ci/sphinxcontrib-jsmath_1611920942228/work
sphinxcontrib-qthelp @ file:///home/ktietz/src/ci/sphinxcontrib-qthelp_1611921055322/work
sphinxcontrib-serializinghtml @ file:///tmp/build/80754af9/sphinxcontrib-serializinghtml_1624451540180/work
spyder @ file:///C:/ci/spyder_1636480369575/work
spyder-kernels @ file:///C:/ci/spyder-kernels_1634237096710/work
SQLAlchemy @ file:///C:/ci/sqlalchemy_1647600017103/work
stack-data @ file:///opt/conda/conda-bld/stack_data_1646927590127/work
statsmodels==0.13.2
sympy @ file:///C:/ci/sympy_1647853873858/work
tables==3.6.1
tabulate==0.8.9
tbb==2021.6.0
tblib @ file:///Users/ktietz/demo/mc3/conda-bld/tblib_1629402031467/work
tenacity @ file:///C:/ci/tenacity_1626248381338/work
terminado @ file:///C:/ci/terminado_1644322780199/work
testpath @ file:///tmp/build/80754af9/testpath_1624638946665/work
text-unidecode @ file:///Users/ktietz/demo/mc3/conda-bld/text-unidecode_1629401354553/work
textdistance @ file:///tmp/build/80754af9/textdistance_1612461398012/work
threadpoolctl @ file:///Users/ktietz/demo/mc3/conda-bld/threadpoolctl_1629802263681/work
three-merge @ file:///tmp/build/80754af9/three-merge_1607553261110/work
tifffile @ file:///tmp/build/80754af9/tifffile_1627275862826/work
tinycss @ file:///tmp/build/80754af9/tinycss_1617713798712/work
tldextract @ file:///opt/conda/conda-bld/tldextract_1646638314385/work
toml @ file:///tmp/build/80754af9/toml_1616166611790/work
tomli @ file:///tmp/build/80754af9/tomli_1637314251069/work
toolz @ file:///tmp/build/80754af9/toolz_1636545406491/work
tornado @ file:///C:/ci/tornado_1606924294691/work
tqdm @ file:///C:/ci/tqdm_1650636210717/work
traitlets @ file:///tmp/build/80754af9/traitlets_1636710298902/work
Twisted @ file:///C:/ci/twisted_1646835413846/work
twisted-iocpsupport @ file:///C:/ci/twisted-iocpsupport_1646798932792/work
typed-ast @ file:///C:/ci/typed-ast_1624953797214/work
typing_extensions @ file:///opt/conda/conda-bld/typing_extensions_1647553014482/work
ujson @ file:///C:/ci/ujson_1648044223886/work
Unidecode @ file:///tmp/build/80754af9/unidecode_1614712377438/work
urllib3 @ file:///C:/ci/urllib3_1650639883891/work
w3lib @ file:///Users/ktietz/demo/mc3/conda-bld/w3lib_1629359764703/work
watchdog @ file:///C:/ci/watchdog_1638367441841/work
wcwidth @ file:///Users/ktietz/demo/mc3/conda-bld/wcwidth_1629357192024/work
webencodings==0.5.1
websocket-client @ file:///C:/ci/websocket-client_1614804375980/work
Werkzeug @ file:///opt/conda/conda-bld/werkzeug_1645628268370/work
widgetsnbextension @ file:///C:/ci/widgetsnbextension_1644991377168/work
win-inet-pton @ file:///C:/ci/win_inet_pton_1605306162074/work
win-unicode-console==0.5
wincertstore==0.2
wrapt @ file:///C:/ci/wrapt_1607574570428/work
xarray==2022.6.0
xarray-einstats==0.3.0
xgboost==1.6.1
xlrd @ file:///tmp/build/80754af9/xlrd_1608072521494/work
XlsxWriter @ file:///opt/conda/conda-bld/xlsxwriter_1649073856329/work
xlwings==0.24.9
xlwt==1.3.0
yapf @ file:///tmp/build/80754af9/yapf_1615749224965/work
yarl==1.8.1
yellowbrick==1.5
zict==2.2.0
zipp==3.8.1
zope.interface @ file:///C:/ci/zope.interface_1625036252485/work

