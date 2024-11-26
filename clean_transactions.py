"""
This module is the initial module used to create the database and initially upload the transaction data into it.
Because Bank of America, American Express, and Venmo all provide different formats for their columns and different
shapes, this provides the necessary functions to standardize all three of the .csv files to have a consistent format
for modelling.

In order to run this module, please upload three .csv files named boa_transactions, amex_transactions, and venmo_transactions
from the respective services.

RUN THIS MODULE FIRST for everything else to work.
"""

# %% import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import sqlite3
import hashlib

boa_filepath = './transactions/boa_transactions'
amex_filepath = './transactions/amex_transactions'
venmo_filepath = './transactions/venmo_transactions'

db_path = 'transactions.db'
conn = sqlite3.connect(db_path)

create_table_query = '''
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    date TEXT,
    description TEXT,
    extended_details TEXT,
    amount TEXT,
    source TEXT,
    merchant TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    category TEXT
)
'''
conn.execute(create_table_query)
conn.commit()

############################################################################################
#                      Read all .csv files in Filepaths                                    #
############################################################################################
def csv_to_df(directory):
    """Return all the files in the filepath as one dataframe"""
    file_names = os.listdir(directory)
    full_file_paths = [os.path.join(directory, file_name) for file_name in file_names]
    dataframes = []

    for csv in full_file_paths:
        if 'venmo' in csv: # venmo csv is shaped differently than the rest
            df = pd.read_csv(csv, header=2)
            df = df.iloc[1:-1,1:]
            df = df.rename(columns={'Amount (total)':'amount'})
        else:
            df = pd.read_csv(csv)
        df.columns = [col.lower().replace(' ',"_") for col in df.columns] # simplify column names
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)

boa_df = csv_to_df(boa_filepath)
amex_df = csv_to_df(amex_filepath)
venmo_df = csv_to_df(venmo_filepath)

############################################################################################
#                      Review and Clean Venmo Data                                         #
############################################################################################

def calc_nan_percent(df):
    """Calculate the % of nan values in each column of a DataFrame"""
    return df.isna().sum(axis=0)/df.shape[0]

# drop amount_(tax), tax_rate because they are all empty strings or no value added
# drop destination and funding_source b/c mostly all null and theres not value added
drop_columns = ['amount_(tax)', 'tax_rate', 'destination',
                'funding_source','destination']

venmo_nan = calc_nan_percent(venmo_df)

# drop columns that have only null values + the list above
venmo_df = venmo_df.drop(columns = list(venmo_nan[venmo_nan == 1].index) + drop_columns)

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

venmo_df[['note','amount']] = venmo_df.apply(extract_amount, axis=1)

# rename columns to match amex data
venmo_df = venmo_df.rename(columns={'datetime':'date',
                                    'note':'description',
                                    'terminal_location':'source'
                                    })

# convert datetime to date object
venmo_df['date'] = pd.to_datetime(venmo_df['date']).dt.date

# create new column to match amex data
venmo_df['extended_details'] = 'from ' + venmo_df['from'] + ' to ' + venmo_df['to'] 

# drop redundant columns
venmo_df = venmo_df.drop(columns=['from','to','type','status','id'])

############################################################################################
#                      Review and Clean BoA Data                                           #
############################################################################################

boa_column_drops = ['split_type', 'user_description','memo','classification',
                    'currency', 'status']
boa_df = boa_df.drop(columns=boa_column_drops) # drop columns with all null values (or only one value)

# convert date string to date object
boa_df['date'] = pd.to_datetime(boa_df['date'], format='%m/%d/%Y').dt.date

# rename columns to match amex data
boa_df = boa_df.rename(columns={
    'original_description':'extended_details',
    'simple_description':'description',
    'account_name':'source'
})

# simplify source column
cond_list = [boa_df['source'].str.contains('Credit Card', na=False),
             boa_df['source'].str.contains('ADV PLUS', na=False)]
value_list = ['boa_credit','boa_debit']
boa_df['source'] = np.select(condlist=cond_list, choicelist=value_list, default='boa_savings')

############################################################################################
#                      Review and Clean AmEx Data                                          #
############################################################################################

amex_df['amount'] = amex_df['amount'] * -1 # make charges negative
amex_df['date'] = pd.to_datetime(amex_df['date'], format='%m/%d/%Y').dt.date # convert to date object
amex_df['source'] = 'amex' # create new column for the source of the data
amex_df = amex_df.rename(columns={'appears_on_your_statement_as':'merchant'}) # convert to simpler name
amex_df[['city','state']] = amex_df['city/state'].str.split('\n', expand=True) # split the city/state column
amex_df = amex_df.drop(columns=['city/state', 'reference']) # drop redundant columns


############################################################################################
#                      Concatenate and Create Unique IDs                                   #
############################################################################################

combined_df = pd.concat([boa_df, amex_df, venmo_df], ignore_index=True)

def create_transaction_id(row):
    """Get unique columns to combine and return a unique id"""
    unique_string = f"{row['date']}_{row['amount']}_{row['extended_details']}_{row['source']}"
    return hashlib.sha256(unique_string.encode()).hexdigest()

combined_df['transaction_id'] = combined_df.apply(create_transaction_id, axis=1)

############################################################################################
#                      Insert into transactions Database                                   #
############################################################################################

def insert_db(df, conn):
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
        cursor.execute(insert_query, (
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
            row['category']
        ))

    conn.commit()

insert_db(combined_df, conn)
conn.close()
