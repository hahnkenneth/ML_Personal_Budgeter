import sqlite3
import pandas as pd

db_path = 'transactions.db'
conn = sqlite3.connect(db_path)

query = '''SELECT * FROM transactions'''
transactions_df = pd.read_sql_query(query, conn)

conn.close()

excel_output_path = 'transactions_for_labelling.xlsx'
transactions_df.to_excel(excel_output_path, index=False)

