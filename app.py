"""
This module is the main app. It is the central location to change between the different pages.

Run this module in order to run the locally hosted Dash app. This will run on 127.0.0.1.
"""

import dash
from dash import dcc, html
from dash import callback, Input, Output
import pandas as pd
import sqlite3  
import dash_bootstrap_components as dbc

# create Dash app with theme
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

from pages import transactions_page
from pages import update_database_page

# overall app layout with the navbar to switch between different pages
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='transactions-store'),  # add this store component to hold data
    dbc.NavbarSimple(
        children=[
            dbc.NavLink("Transaction Dashboard", href="/", active="exact"),
            dbc.NavLink("Update Database", href="/update-database", active="exact"),
            dbc.NavLink("Train Model", href="/train-model", active="exact")
        ],
        brand="Dashboard Navigation",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    dash.page_container  # container for different pages
], fluid=True)

@callback(
    Output('transactions-store', 'data'),
    Input('url', 'pathname')  # triggered when the page changes
)
def load_data(_):
    """callback function to store transaction data from sqlite database ahead of time.
    The function reupdates with every page change"""

    # connect to sqlite database
    db_path = 'transactions.db'
    conn = sqlite3.connect(db_path)

    # get all transaction data and store as a DataFrame
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC", conn)
    conn.close()

    return df.to_dict('records')

if __name__ == "__main__":
    app.run_server(debug=True)