"""
This page on the dash app is a dashboard to see the most important data from the 
transactions database. The user will be able to see the amount spent and amount gained
on a timetrend. On top of this, they will be able to see the TotalAmount spent for each
source (BoA, AmEx, or Venmo). Finally, they can review the pie chart to see the % spent
by each category.

The sidebar is a series of filters to review the data for better granularity.
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import train_model
from dash import dash_table
import pandas as pd
import plotly.express as px
import sqlite3
import joblib
import os
from custom_preprocessors import (
    DateFeatureProcessor, 
    CyclicalEncoder,
    TextPreprocessor,
    MultiColumnOneHotEncoder,
    ColumnScaler
)

# register Dash app page
dash.register_page(__name__, path='/')

# sidebar layout with dbc components (used for filtering the data)
sidebar = dbc.Col(
            [
                html.H2("Filters", className="display-6"), # title
                html.Hr(),
                html.P("Filter transaction data:", className="lead"), # header
                
                # select the time aggregation via dropdown
                html.Label('Aggregation Level', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='aggregation-level',
                    options=[
                        {'label': 'Daily', 'value': 'day'},
                        {'label': 'Weekly', 'value': 'week'},
                        {'label': 'Monthly', 'value': 'month'},
                        {'label': 'Yearly', 'value': 'year'}
                    ],
                    value='month',
                    placeholder='Select Aggregation Level',
                    style={'margin-bottom': '10px'}
                ),

                # select whther to look at the true labels or the predicted labels by the model
                html.Label('Category Labels Type', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='label-type',
                    options=[
                        {'label': 'Predicted Category', 'value': 'predicted_labels'},
                        {'label': 'True Category', 'value': 'labels'}
                    ],
                    value='predicted_labels',
                    placeholder='Select Category Label Type',
                    style={'margin-bottom': '10px'}
                ),

                # select which categories to display
                html.Label('Category Filter', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='category-filter',
                    multi=True,
                    placeholder='Select Categories to filter by',
                    style={'margin-bottom': '10px'}
                ),

                # select the time period to display the data
                html.Label('Choose Date Range', style={'font-weight': 'bold'}),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    display_format='YYYY-MM-DD',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    style={'margin-bottom': '20px'}
                )
            ],
            width=2,
            style={
                'padding': '20px',
                'background-color': '#f8f9fa',
                'height': '100vh',
                'position': 'fixed'
            }
        )

# content layout using dbc components
content = dbc.Col(
    [
        html.H1("Transaction Data by Category Dashboard", className='mt-4 mb-4'), # title
        dcc.Graph(id='time-trend-graph-spent'), # time trend of spending
        dcc.Graph(id='time-trend-graph-gained'), # time trend of income
        dbc.Row([
            dbc.Col(dcc.Graph(id='source-bar-chart'), width=6), # spending by source (bar chart)
            dbc.Col(dcc.Graph(id='category-pie-chart'), width=6) # spending by category (pie chart)
        ], className='mt-4'),
        dbc.Row([
            dbc.Col(
                # view all transaction data in a DataTable
                dash_table.DataTable(
                    id='transactions-table-transactions',
                    style_table={'overflowX': 'auto'},
                    page_size=10,
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'}
                ), className='mt-4'
            )
        ])
    ],
    width=9,
    style={
        'margin-left': '320px',
        'padding': '20px'
    }
)

# main layout to visualize and filter transaction data
layout = dbc.Container(
    [
        dbc.Row([
            sidebar,
            content
        ])
    ],
    fluid=True
)

@callback(
    [Output('category-filter', 'options'),
     Output('time-trend-graph-spent', 'figure'),
     Output('time-trend-graph-gained', 'figure'),
     Output('source-bar-chart', 'figure'),
     Output('category-pie-chart', 'figure'),
     Output('transactions-table-transactions', 'data')],
    [Input('aggregation-level', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('label-type', 'value'),
     Input('category-filter', 'value'),
     Input('transactions-store', 'data')]
)
def update_dash_plots(aggregation_level, start_date, end_date, label_type, selected_categories, store_data):
    """This callback function is for the upload csv button that the user can upload a csv file with 
    labelled rows. The function will look at the transaction_id and update that row.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - aggregation_level: the time aggregation selection filter for transaction data
    - start_date: beginning date to view transactions
    - end_date: end date to view transactions
    - label_type: whether to see predicted or true labels on the plot
    - selected_categories: which transaction categories to view
    - store_data: transaction database data

    Output:
    - category-filter: informs the dropdown mention which categories are already selected to view
    - time-trend-graph-spent: time trend of amount spent by category
    - time-trend-graph-gained: time trend of amount gained by category
    - source-bar-chart: bar chart of amount spent + gained by source (AmEx, BoA, or Venmo)
    - category-pie-chart: pie chart of categories
    - transactions-table-transactions: the DataTable of transaction data
    """

    # convert transaction data to DataFrame and preprocess
    df = pd.DataFrame(store_data)
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(str).str.replace(',', '').astype(float) 

    # filter by date range
    if start_date and end_date:
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # commonize the labels
    df['labels'] = df['labels'].str.capitalize().str.replace('_', ' ')
    df['predicted_labels'] = df['predicted_labels'].str.capitalize().str.replace('_', ' ')

    # get unique categories for the category dropdown filter
    unique_categories = df['labels'].dropna().unique()
    category_options = [{'label': category, 'value': category} for category in unique_categories]

    # if there are no categories selected, select all categories
    if not selected_categories:
        selected_categories = unique_categories.tolist()
    
    # get all the rows that are in the selected categories filter
    df = df[df[label_type].isin(selected_categories)]

    # separate by positive and negative transactions for amount gained and spent, respectively
    df_pos = df[df['amount'] > 0]
    df_neg = df[df['amount'] < 0]
    df_list = [df_pos, df_neg]
    df_agg_list = [df_pos, df_neg]

    # aggregate the data based on month, day, year, or week from the aggregation filter
    for i, df_i in enumerate(df_list):
        # aggregation logic
        if aggregation_level == 'month':
            df_i['month'] = df_i['date'].dt.month
            df_i['year'] = df_i['date'].dt.year
            df_i['month-year'] = df_i['year'].astype(str) + '-' + df_i['month'].astype(str)
            df_agg_list[i] = df_i.groupby(['month-year', 'source', label_type])['amount'].sum().reset_index()
            df_agg_list[i]['date'] = pd.to_datetime(df_agg_list[i]['month-year'], format='%Y-%m').dt.date
        elif aggregation_level == 'year':
            df_i['year'] = df_i['date'].dt.year
            df_agg_list[i] = df_i.groupby(['year', 'source', label_type])['amount'].sum().reset_index()
            df_agg_list[i].rename(columns={'year': 'date'}, inplace=True)
        elif aggregation_level == 'day':
            df_i['day'] = df_i['date'].dt.date
            df_agg_list[i] = df_i.groupby(['day', 'source', label_type])['amount'].sum().reset_index()
            df_agg_list[i].rename(columns={'day': 'date'}, inplace=True)
        elif aggregation_level == 'week':
            df_i['week'] = df_i['date'].dt.isocalendar().week
            df_i['year'] = df_i['date'].dt.year
            df_i['week-year'] = df_i['year'].astype(str) + '-W' + df_i['week'].astype(str) + '-1'
            df_agg_list[i] = df_i.groupby(['week-year', 'source', label_type])['amount'].sum().reset_index()
            df_agg_list[i]['date'] = pd.to_datetime(df_agg_list[i]['week-year'], format='%Y-W%W-%w').dt.date
        else:
            df_agg_list[i] = df_i
    
    # separate the list into positive and negative again
    df_agg_pos = df_agg_list[0]
    df_agg_neg = df_agg_list[1]

    # reconcatenate these for the source bar chart
    df_agg = pd.concat([df_agg_pos, df_agg_neg])

    # change the source names to more proper names
    custom_labels = {
        'Venmo': 'Venmo',
        'amex': 'AmEx',
        'boa_credit': 'BoA Credit',
        'boa_debit': 'BoA Debit'
    }

    # group by source for the amount column
    df_source_agg = df_agg.groupby('source')['amount'].sum().reset_index()

    # change the source names
    df_source_agg['source'] = df_source_agg['source'].map(custom_labels).fillna(df_source_agg['source'])
    df_source_agg = df_source_agg[df_source_agg['source'] != 'boa_savings']

    # create the bar figure to see the aggregation by source
    bar_fig = px.bar(
        df_source_agg,
        x='source',
        y='amount',
        title='Total Amount by Account',
        color='source',
        labels={'source': 'Account', 'amount': 'Total Amount'}
    )

    # modify the bar figure
    bar_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )

    # determine whether the user wants to see predicted categories or true categories
    # change the title of the plots based on this selection
    if label_type == 'predicted_labels':
        title_str = 'Predicted Categories'
    else:
        title_str = 'True Categories'

    # groupby predicted_labels or labels based on the selection
    df_pie = df_agg_neg.groupby(label_type)['amount'].sum().reset_index()
    df_pie['amount'] = df_pie['amount'].abs()

    # create the pie chart by category
    pie_fig = px.pie(
        df_pie,
        names=label_type,
        values='amount',
        title=f'Percentage of Amount Spent per Category by {title_str}',
        labels={label_type: 'Category', 'amount': 'Total Negative Amount'}
    )

    # modify the pie chart
    pie_fig.update_layout(
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)'
    )

    # create the time trend for spending transactions (color by the categories)
    # category can either be predicted category or true category
    neg_area_fig = px.line(
        df_agg_neg,
        x='date',
        y='amount',
        color=label_type,
        title=f'Amount Spent Over Time by {title_str}',
        labels={'date': 'Date', 'amount': 'Amount', label_type: 'Category'},
    )

    neg_area_fig.update_layout(
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    xaxis=dict(showgrid=True, gridcolor='lightgrey'), 
    yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )

    # create the time trend for gained transactions (color by the categories)
    # categories can either be predicted by the model or the true category
    pos_area_fig = px.line(
        df_agg_pos,
        x='date',
        y='amount',
        color=label_type,
        title=f'Amount Gained Over Time by {title_str}',
        labels={'date': 'Date', 'amount': 'Amount', label_type: 'Category'},
    )

    pos_area_fig.update_layout(
    plot_bgcolor='rgba(0, 0, 0, 0)', 
    paper_bgcolor='rgba(0, 0, 0, 0)', 
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )

    # convert to dictionary for the data_table
    table_data = df.to_dict('records')

    return category_options, neg_area_fig, pos_area_fig, bar_fig, pie_fig, table_data

@callback(
    [
        Output('date-picker-range', 'start_date'),
        Output('date-picker-range', 'end_date'),
        Output('date-picker-range', 'min_date_allowed'),
        Output('date-picker-range', 'max_date_allowed')
    ],
    Input('transactions-store', 'data')
)
def update_date_picker_range(store_data):
    """This callback function is update the date picker range based on the user's selection. This function
    also limits the max and min dates based on the transaction data so users cannot select dates beyond the
    dates of the transaction.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - store_data: transaction database data

    Output:
    - start_date: default start date
    - end_date: default end date
    - min_date_allowed: the minimum date available in the transaction data
    - max_date_allowed: the maximum date available in the transaction data
    """

    if store_data is None:
        raise dash.exceptions.PreventUpdate

    # convert the data to a DataFrame
    df = pd.DataFrame(store_data)

    # ensure the date column is of datetime type
    df['date'] = pd.to_datetime(df['date'])

    # get the min and max dates
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    # set default start and end dates to be the min and max dates
    start_date = min_date
    end_date = max_date

    return start_date, end_date, min_date, max_date