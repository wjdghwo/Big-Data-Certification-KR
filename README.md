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

df = data.reset_index().melt(id_vars=['index','channel'], var_name = ['date']).rename(columns = {'index':'type'})



# 왼쪽 그래프 평균값 비교 비율 막대그래프
# 유형 비교 선택지 추가, 선택시 오른쪽 그래프 값 변경

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


def create_time_series(dff, axis_type, title):
    fig = px.scatter(dff, x='date', y='value')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes()
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

    
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
                'first channel',
                id='crossfilter-xaxis-column',
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df['channel'].unique(),
                'second channel',
                id='crossfilter-yaxis-column'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
        
        html.Div([
            dcc.RadioItems(
                df['type'].unique(),
                '취급건수',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )])
    ], style={
        'padding': '10px 5px'
    }),
    
    # 우측상단 그래프
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'type'}]}
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
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-date--slider', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name, yaxis_type, date_value):
    dff = df[df['date'] == date_value]
    dff1 = dff.groupby(['type','channel']).mean().reset_index()
    fig = px.histogram(dff1[(dff1['channel'] == xaxis_column_name)|(dff1['channel'] == yaxis_column_name)], 
                       x='type', y='value', color = 'channel', barnorm = "percent")    
    fig.update_traces()
    fig.update_xaxes()
    fig.update_yaxes()
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig

@app.callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['type'] == axis_type]
    dff = dff[dff['channel'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)

@app.callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['type'] == axis_type]
    dff = dff[dff['channel'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)

if __name__ == '__main__':
    app.run_server(host='localhost',port=8007)











































# 왼쪽 그래프 산점도 
# 클릭하면 오른쪽 그래프 값 



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


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
def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, date_value):
    dff = df[df['date'] == date_value]
    
    fig = px.scatter(x=dff[dff['channel'] == xaxis_column_name]['value'],
                     y=dff[dff['channel'] == yaxis_column_name]['value'],
                     hover_name=dff[dff['channel'] == yaxis_column_name]['type'])
    
    fig.update_traces(customdata=dff[dff['channel'] == yaxis_column_name]['type'])
    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
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





































absl-py==1.2.0
alembic==1.8.1
altair==4.2.0
astor==0.8.1
asttokens==2.0.8
astunparse==1.6.3
attrs==22.1.0
autograd==1.4
autograd-gamma==0.5.0
autopage==0.5.1
backcall==0.2.0
bayesian-optimization==1.2.0
beautifulsoup4==4.11.1
bob==11.1.1
Brotli==1.0.9
cachetools==5.2.0
catboost==1.0.6
category-encoders==2.5.0
certifi==2022.6.15
charset-normalizer==2.1.1
click==8.1.3
cliff==4.0.0
cmaes==0.8.2
cmd2==2.4.2
colorama==0.4.5
colorlog==6.6.0
cycler==0.11.0
Cython==0.29.28
dash==2.6.1
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
decorator==5.1.1
distlib==0.3.5
eli5==0.13.0
entrypoints==0.4
ephem==4.1.3
executing==0.10.0
filelock==3.8.0
Flask==2.2.2
Flask-Compress==1.12
flatbuffers==1.12
fonttools==4.36.0
formulaic==0.4.0
future==0.18.2
gast==0.4.0
gensim==4.2.0
google==3.0.0
google-api-core==2.8.2
google-auth==2.11.0
google-auth-oauthlib==0.4.6
google-cloud==0.34.0
google-cloud-automl==2.8.1
google-pasta==0.2.0
googleapis-common-protos==1.56.4
graphviz==0.20.1
greenlet==1.1.2
grpcio==1.47.0
grpcio-status==1.47.0
h5py==3.7.0
huggingface-hub==0.8.1
idna==3.3
importlib-metadata==4.12.0
interface-meta==1.3.0
ipython==8.4.0
itsdangerous==2.1.2
jedi==0.18.1
Jinja2==3.1.2
joblib==1.1.0
jsonschema==4.14.0
keras==2.9.0
Keras-Preprocessing==1.1.2
kiwisolver==1.4.4
korean-lunar-calendar==0.2.1
libclang==14.0.6
lifelines==0.27.1
lightgbm==3.3.2
llvmlite==0.39.0
Mako==1.2.1
Markdown==3.4.1
MarkupSafe==2.1.1
matplotlib==3.5.3
matplotlib-inline==0.1.6
missingno==0.5.1
missingpy==0.2.0
mpmath==1.2.1
networkx==2.8.6
ngboost==0.3.12
numba==0.56.0
numpy==1.22.4
oauthlib==3.2.0
opt-einsum==3.3.0
optuna==2.10.1
packaging==21.3
pandas==1.4.3
parso==0.8.3
patsy==0.5.2
pbr==5.10.0
pickleshare==0.7.5
Pillow==9.2.0
pipeline==0.1.0
pipenv==2022.8.19
platformdirs==2.5.2
plotly @ file:///C:/Users/Administrator/Downloads/plotly-5.10.0-py2.py3-none-any.whl
prettytable==3.3.0
prompt-toolkit==3.0.30
proto-plus==1.22.0
protobuf==3.19.4
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
PyBrain==0.3
pydot==1.4.2
Pygments==2.13.0
PyMeeus==0.5.11
PyMySQL==1.0.2
pyod==1.0.4
pyparsing==3.0.9
pyperclip==1.8.2
pyreadline3==3.4.1
pyrsistent==0.18.1
python-dateutil==2.8.2
python-math==0.0.1
pytz==2022.2.1
PyYAML==6.0
ramp==0.1.4
regex==2022.8.17
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
scikit-learn==1.1.2
scipy==1.9.0
seaborn==0.11.2
setuptools-git==1.2
six==1.16.0
slicer==0.0.7
smart-open==6.1.0
soupsieve==2.3.2.post1
SQLAlchemy==1.4.40
stack-data==0.4.0
stats==0.1.2a0
statsmodels==0.13.2
stevedore==4.0.0
sympy==1.10.1
tabulate==0.8.10
tenacity==8.0.1
tensorboard==2.9.1
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.9.1
tensorflow-estimator==2.9.0
tensorflow-io-gcs-filesystem==0.26.0
termcolor==1.1.0
Theano==1.0.5
threadpoolctl==3.1.0
tokenizers==0.12.1
toolz==0.12.0
torch==1.12.1
tqdm==4.64.0
traitlets==5.3.0
transformers==4.21.1
typing_extensions==4.3.0
urllib3==1.26.12
virtualenv==20.16.3
virtualenv-clone==0.5.7
wcwidth==0.2.5
Werkzeug==2.2.2
wrapt==1.14.1
xgboost==1.6.2
zipp==3.8.1
