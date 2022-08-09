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


