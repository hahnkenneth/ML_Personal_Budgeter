# ML_Personal_Budgeter
Dash App project to compile my transaction data from various institutions. An ML model is used to predict the categories of the transactions and displays the data on the app.  

To review the overview of the project and how to use the Dash app, please refer to my [projects portfolio page](https://hahnkenneth.github.io/projects/6_project/).  

To run this for yourself, please download detailed csv statements from Bank of America, Venmo, and American Express. Store these in the \transactions directory. Then run the clean_transactions.py file once. This will create the SQLite database, clean the transactions, and upload them to the database.  

Next, review the model_selection_notebook.ipynb in order to determine see my justification behind the feature engineering involved and the model I chose to use for classification. I ultimately ended on a ensemble soft-voting model comprised of a Logistic Regression, SVC, and Neural Network model that resulted in a 90% test accuracy.  

The penultimate step is to run the custom_preprocessors.py file in order to save a .pkl file that is the Pipeline for preprocessing the transaction information. Finally, once that is completed (and it only needs to be completed once), you can then just run the app.py and view the app locally to have your own custom categorizer.
