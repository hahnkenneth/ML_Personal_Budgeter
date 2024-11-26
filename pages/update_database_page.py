"""
This page will be used to update new data into the database. The user will be able to see the
contents of the transactions database. If there is new statement data from any of the three
institutions (BoA, AmEx, or Venmo), the user can upload this data and it will upload into the
database. The user can then use the CustomModel in order to predict the categories of the new 
transactions. 

The user also will have access to be able to download unlabelled data in the form of a .csv file.
This will allow the user to label the transaction data in order to retrain the model (if needed).
Once the user has labelled the new data, they can upload the labelled data in the form of a .csv
file.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import train_model
from dash import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import joblib
import os
import re
import hashlib
import base64
import io
from custom_preprocessors import (
    DateFeatureProcessor, 
    CyclicalEncoder,
    TextPreprocessor,
    MultiColumnOneHotEncoder,
    ColumnScaler
)
import train_model
from train_model import CustomModel

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from custom_preprocessors import (
    DateFeatureProcessor, 
    CyclicalEncoder,
    TextPreprocessor,
    MultiColumnOneHotEncoder,
    ColumnScaler
) 
 
from sklearn.preprocessing import (
    LabelBinarizer
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# register Dash app page
dash.register_page(__name__, path='/update-database')

# sidebar layout with dbc components
sidebar = dbc.Col(
            [
                html.H2("Uploads", className="display-6"), # sidebar title
                html.Hr(),
                html.P("Upload Bank Statements", className="lead"), # header
                
                html.Label('Upload Bank of America CSV', style={'font-weight': 'bold'}), # header
                
                # upload function for Bank of America transaction .csv files
                dcc.Upload(
                    id='boa-upload',
                    children=html.Div(['Drag and Drop or ', html.A('Select BoA CSV')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-bottom': '20px'
                    }
                ),
                
                # upload function for American Express transaction .csv files
                html.Label('Upload Amex CSV', style={'font-weight': 'bold'}),
                dcc.Upload(
                    id='amex-upload',
                    children=html.Div(['Drag and Drop or ', html.A('Select Amex CSV')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-bottom': '20px'
                    },
                    multiple=False
                ),        

                # upload function for Venmo transaction .csv files
                html.Label('Upload Venmo CSV', style={'font-weight': 'bold'}),
                dcc.Upload(
                    id='venmo-upload',
                    children=html.Div(['Drag and Drop or ', html.A('Select Venmo CSV')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-bottom': '20px'
                    },
                    multiple=False
                ),
                # button to upload data into the database
                dbc.Button("Upload Data", id='train-button', color='primary', className="mt-3"),

                html.Div(id='output-message-upload', className='mt-4'), # upload message 

                # button to predict the category classes once the data is uploaded
                dbc.Button("Predict Categories", id='predict-button', color='success', className="mt-3"),

                html.Div(id='output-message-predict', className='mt-4')
            ],
            width=2,
            style={
                'padding': '20px',
                'background-color': '#f8f9fa',
                'height': '100vh',
                'position': 'fixed'
            }
        )

# content layout to show the transactions DataTable and the download and upload the DataTable
content = html.Div([dbc.Col(
        [
            html.H1("Transaction Database", className='mt-4 mb-4'), # title
            dbc.Row(
                [
                    # button to upload labelled data (used for training the model)
                    dbc.Col(
                        dcc.Upload(
                            id='upload-labelled-csv-button',
                            children=dbc.Button("Upload Labelled Data CSV", color="secondary"),
                            style={
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        width='auto'
                    ),

                    # button to download unlabelled data
                    dbc.Col(
                        dbc.Button("Download Unlabelled Data CSV", color="primary", id="download-unlabelled-csv-button", className='mr-2'),
                        width='auto'
                    ),
                    dcc.Download(id='download-labelled-dataframe-csv')
                ],
                align="center",
                justify="start",
                style={'margin-bottom':'10px'}
            ),
            # message for successful upload
            html.Div(id='output-upload-message'),

            # display the transactions database in a DataTable
            dash_table.DataTable(
                id='transactions-table-update-database',
                style_table={'overflowX': 'scroll'},
                style_cell={
                    'minWidth': '80px',
                    'maxWidth': '150px',
                    'whiteSpace': 'normal'
                },
                page_size=50,
                editable=True,
                filter_action="native",
                sort_action='native',
                sort_mode='multi',
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'height': 'auto'},
            )
        ],
        width=9,
        style={
            'margin-left': '320px',
            'padding': '20px'
        }
    ),
    html.Div(id='table-dropdown-container')
])

# main layout using dbc.Container, dbc.Row, and dbc.Col
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
    Output('output-message-upload', 'children'),
    Input('train-button', 'n_clicks'),
    [State('boa-upload', 'contents'),
     State('boa-upload', 'filename'),
     State('amex-upload', 'contents'),
     State('amex-upload', 'filename'),
     State('venmo-upload', 'contents'),
     State('venmo-upload', 'filename')]
)
def upload_txns(n_clicks, boa_contents, boa_filename, amex_contents, amex_filename, venmo_contents, venmo_filename):
    """The callback function used in the sidebar to upload the Bank of America, American Express, and Venmo
    transactions into the transactions database
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - n_clicks: Calls the function and begins uploading the csv files
    - boa_contents: transaction data from Bank of America in csv format
    - boa_filename: file name of the csv file
    - amex_contents: transaction data from American Express in csv format
    - amex_filename: file name of the csv file
    - venmo_contents: transaction data from Venmo in csv format
    - venmo_filename: file name of the csv file

    Output:
    - output-message-upload: message to the user when the file(s) are successfully uploaded
    """
    if n_clicks is None:
        return "Upload the files and click Upload Data to start."

    # create a connection to the database
    db_path = 'transactions.db'
    conn = sqlite3.connect(db_path)

    def process_and_insert(contents, filename, source_type):
        """decodes the files use b64 decode and then process the data using the bank specific
        processing function. This will then standardize the data to have the same features and
        format and finally, the insert_db will upload the data.
        ---------------------------------------------------------------------------------------
        Inputs:
        contents: the file contents
        filename: the name of the file
        source-type: a string that is either 'venmo', 'boa', or 'amex' used for processing the
        specific bank transaction data.

        Outputs:
        num_txns: a value used to inform the user how many transactions were stored in the database.
        """
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            file = io.StringIO(decoded.decode('utf-8'))

            if source_type == 'venmo':
                df, num_txns = process_venmo_data(file)
            elif source_type == 'boa':
                df, num_txns = process_boa_data(file)
            elif source_type == 'amex':
                df, num_txns = process_amex_data(file)
            
            insert_db(df, conn)
            return num_txns
        return 0
    
    # process and insert the banks transaction csv files
    num_txns_b = process_and_insert(boa_contents, boa_filename, 'boa')
    num_txns_a = process_and_insert(amex_contents, amex_filename, 'amex')
    num_txns_v = process_and_insert(venmo_contents, venmo_filename, 'venmo')

    # calculate the total number of transactions uploaded
    total_txns = num_txns_v + num_txns_b + num_txns_a

    # close connection to the database
    conn.close()

    # return a message based on whether a file was successfully loaded or not
    if total_txns > 0:
        return f"Files successfully uploaded and {total_txns} transactions inserted into the database"
    else:
        return "No files were uploaded. Please upload at least one file to continue."

def process_venmo_data(file):
    """The function takes in a venmo csv file and processes it to have a standardized number of
    features and format."""

    # read csv file and only select the specific columns and rows (the excluded ones are extra whitespace)
    df = pd.read_csv(file, header=2)
    df = df.iloc[1:-1,1:]

    # rename to standardize the amount column name
    df = df.rename(columns={'Amount (total)':'amount'})

    df.columns = [col.lower().replace(' ',"_") for col in df.columns] # simplify column names

    # drop amount_(tax), tax_rate because they are all empty strings or no value added
    # drop destination and funding_source b/c mostly all null and theres not value added
    drop_columns = ['amount_(tax)', 'tax_rate', 'destination',
                    'funding_source','destination']

    venmo_nan = df.isna().sum(axis=0)/df.shape[0]

    # drop columns that have only null values + the list above
    df = df.drop(columns = list(venmo_nan[venmo_nan == 1].index) + drop_columns)

    def extract_amount(row):
        """regex to extract the amount as a float instead of a string.
        Also convert datetime string to a date object"""
        # delete commas, parenthesis and whitespace
        amount = float(re.sub(r'\$|,|\(|\)| ','',row['amount']))
        
        # check if I paid the person, then make it negative
        if row['type'] == 'Payment' and row['from'] == 'Kenneth Hahn':
            amount = -1*abs(amount)

        # if it's a transfer to my bank, then make the note the Standard Transfer type (description)
        note = row['note']
        if row['type'] == 'Standard Transfer':
            note = 'Standard Transfer'

        return pd.Series([note, amount], index=['note', 'amount'])

    df[['note','amount']] = df.apply(extract_amount, axis=1)

    # rename columns to match amex data
    df = df.rename(columns={'datetime':'date',
                                        'note':'description',
                                        'terminal_location':'source'
                                        })

    # convert datetime to date object
    df['date'] = pd.to_datetime(df['date']).dt.date

    # create new column to match amex data
    df['extended_details'] = 'from ' + df['from'] + ' to ' + df['to'] 

    # drop redundant columns
    df = df.drop(columns=['from','to','type','status','id'])
    num_txns = df.shape[0]

    df['merchant'] = ''
    df['address'] = ''
    df['city'] = ''
    df['state'] = ''
    df['zip_code'] = None
    df['category'] = ''

    return df, num_txns

def process_boa_data(file):
    """The function takes in a Bank of America csv file and processes it to have a standardized number of
    features and format."""

    # read decoded csv file
    df = pd.read_csv(file)
    df.columns = [col.lower().replace(' ',"_") for col in df.columns] # simplify column names

    boa_column_drops = ['split_type', 'user_description','memo','classification',
                    'currency', 'status']
    df = df.drop(columns=boa_column_drops) # drop columns with all null values (or only one value)

    # convert date string to date object
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.date

    # rename columns to match amex data
    df = df.rename(columns={
        'original_description':'extended_details',
        'simple_description':'description',
        'account_name':'source'
    })

    # simplify source column
    cond_list = [df['source'].str.contains('Credit Card', na=False),
                df['source'].str.contains('ADV PLUS', na=False)]
    value_list = ['boa_credit','boa_debit']
    df['source'] = np.select(condlist=cond_list, choicelist=value_list, default='boa_savings')
    num_txns = df.shape[0]
    df['merchant'] = ''
    df['address'] = ''
    df['city'] = ''
    df['state'] = ''
    df['zip_code'] = None
    
    print(f'BoA Columns: {df.columns}')

    return df, num_txns

def process_amex_data(file):
    """The function takes in an American Express csv file and processes it to have a standardized number of
    features and format."""

    # read amex csv file
    df = pd.read_csv(file)
    df.columns = [col.lower().replace(' ',"_") for col in df.columns] # simplify column names

    df['amount'] = df['amount'] * -1 # make charges negative
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.date # convert to date object
    df['source'] = 'amex' # create new column for the source of the data
    df = df.rename(columns={'appears_on_your_statement_as':'merchant'}) # convert to simpler name
    df[['city','state']] = df['city/state'].str.split('\n', expand=True) # split the city/state column
    df = df.drop(columns=['city/state', 'reference']) # drop redundant columns
    num_txns = df.shape[0]

    return df, num_txns

def insert_db(df, conn):
    """This function will take a DataFrame and upload it into the transactions database"""

    cursor = conn.cursor()
    insert_query = '''
    INSERT OR IGNORE INTO transactions (
        transaction_id,
        date,
        description,
        extended_details,
        amount,
        source,
        merchant,
        address,
        city,
        state,
        zip_code,
        category
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    for _, row in df.iterrows():
        # create a unique sha256 index for each transaction using the date, amount, extra details, and source
        transaction_id = hashlib.sha256(f"{row['date']}_{row['amount']}_{row['extended_details']}_{row['source']}".encode()).hexdigest()

        # upload all other data into the database
        cursor.execute(insert_query, (
            transaction_id,
            row['date'],
            row['description'],
            row['extended_details'],
            row['amount'],
            row['source'],
            row['merchant'],
            row['address'],
            row['city'],
            row['state'],
            row['zip_code'],
            row['category']
        ))

    conn.commit()

# load the tokenizer and embedding model
tokenizer, embed_model = train_model.load_embed_model()

# load the Custom Model to be used for prediction
model_path = './models'
custom_model = CustomModel.load(model_path)

@callback(
    Output('output-message-predict', 'children'),
    Input('predict-button', 'n_clicks')
)
def predict_categories(n_clicks):
    """The callback function will predict the transaction category for all unpredicted rows once the 
    button is clicked
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - n_clicks: Calls the function and begins making predictions for unlabelled transactions

    Output:
    - output-message-predict: Message to inform the user when all transactions have finished prediction
    """
    if n_clicks is None:
        return "Click the Predict button to start the prediction."
    
    print('Starting Full Code...')

    # connect to database and get all transactions where the model has not predicted the category
    db_path = 'transactions.db'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM transactions WHERE  predicted_labels NULL", conn)

    # extract_and_preprocess_txns will provide the X data used for predictions
    # _ is a variable that is not used for predictions but is used for training a model
    _, X = train_model.extract_and_preprocess_txns(df=df, db_filepath=None, train_or_predict='predict')


    # load the pipeline to transform the X data
    pipeline = joblib.load('preprocess_pipeline_fitted.pkl')
    X_processed = pipeline.transform(X)

    # drop the predicted_labels as it is not used for predictions
    X_processed = X_processed.drop(columns='predicted_labels')

    print('Starting Embedding...')
    # create embeddings of the extended_text column (created from the pipeline)
    X_embed = train_model.embed_text(X_processed['extended_text'], tokenizer, embed_model)
    print('Embedding Completed')

    # Convert the X_data into other_features and embeddings (other features is all features except the text data)
    # also keep the id column as a separate array and remove it from the X data
    X_other_features, X_embeddings, X_transaction_ids = train_model.embedding_otherfeatures_split(X_processed, X_embed)

    print('Starting Predictions...')

    # use the model to create prediction
    predictions = custom_model.predict(X_embeddings, X_other_features)

    # class array for predictions
    classes = np.array(['debt', 'dining', 'education', 'entertainment', 'fees',
       'groceries', 'healthcare', 'housing', 'ignore', 'income',
       'insurance', 'investments', 'misc', 'personal_care', 'shopping',
       'subscriptions', 'transportation', 'travel', 'utilities'])

    # convert the predictions to class string labels
    predicted_labels = classes[predictions]

    print('Predictions Completed')

    # store the predictions into the database
    train_model.store_predictions(db_path, X_transaction_ids, predicted_labels)

    return "Predictions Stored Successfully"

@callback(
    [Output('transactions-table-update-database', 'data'),
     Output('transactions-table-update-database', 'columns')],
    Input('transactions-store', 'data')
)
def update_transactions_table(store_data):
    """This callback function is to refresh the DataTable displayed on the main content of the page
    whenever the page refreshes.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - store_data: the transactions-store data from the transactions database

    Output:
    - transactions-table-update-database: the data and the columns to be displayed in the DataTable.
    """
    if store_data is None:
        raise dash.exceptions.PreventUpdate

    # convert the store_data to a DataFrame and extract the data and the columns
    df = pd.DataFrame(store_data)
    data = df.to_dict('records')
    columns = [{"name": i, "id": i} for i in df.columns]

    return data, columns

@callback(
    Output('download-labelled-dataframe-csv', 'data'),
    Input("download-unlabelled-csv-button", 'n_clicks'),
    State('transactions-table-update-database', 'data'),
    prevent_initial_call=True
)
def download_csv(n_clicks, table_data):
    """This callback function is for the download_csv button that will download the transactions database
    as a .csv file. The file will only have rows that are unlabelled for the user to label them and upload
    to the database. This is only used when wanting to update the accuracy numbers or to retrain the model.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - n_clicks: used for whenever the user clicks on the download button
    
    State:
    - table_data: the transactions-store data from the transactions database

    Output:
    - download-labelled-dataframe-csv: the .csv file to be downloaded with unlabelled transactions
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # convert data to DataFrame and only get the rows that are unlabelled
    df = pd.DataFrame(table_data)
    df = df[df['labels'].isna()]

    # return the df as a .csv
    return dcc.send_data_frame(df.to_csv, "unlabelled_transactions.csv", index=False)


@callback(
    Output("output-upload-message", "children"),
    Input('upload-labelled-csv-button', 'contents'),
    State('upload-labelled-csv-button', 'filename'),
    prevent_initial_call=True
)
def upload_labelled_csv(contents, filename):
    """This callback function is for the upload csv button that the user can upload a csv file with 
    labelled rows. The function will look at the transaction_id and update that row.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - contents: the .csv file contents
    - filename: the name of the .csv file

    Output:
    - output-upload-message: informs the user on the status of the upload
    """

    if contents is None:
        raise dash.exceptions.PreventUpdate
    
    # split the csv file and decode the .csv file
    content_type, content_str = contents.split(',')
    decoded = base64.b64decode(content_str)
    
    try:
        # read csv and process data
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # connect to sqlite database
        db_path = 'transactions.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # update each of the rows in the transactions database
        for _, row in df.iterrows():
            upload_query = '''
            INSERT OR REPLACE INTO transactions (
                transaction_id,
                date,
                description,
                extended_details,
                amount,
                source,
                merchant,
                address,
                city,
                state,
                zip_code,
                category,
                labels,
                predicted_labels
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            '''
            cursor.execute(upload_query, (
                row['transaction_id'],
                row['date'],
                row['description'],
                row['extended_details'],
                row['amount'],
                row['source'],
                row['merchant'],
                row['address'],
                row['city'],
                row['state'],
                row['zip_code'],
                row['category'],
                row['labels'],
                row['predicted_labels']
            ))
        conn.commit()
        conn.close()
        
        # output message
        return f"File {filename} successfully uploaded and transactions uploaded and transactions updated."
    except Exception as e:
        return f"There was an error processesing the file '{filename}': {str(e)}"

