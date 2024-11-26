"""
This module is the front end for the model training page. This page will allow the user
to review the current accuracy, loss, and confusion matrix of the trained model. With more data,
the user will be able to retrain the model from this page and review the new accuracy, loss, and
confusion matrix next to the old model. If the user is satisfied with the new metrics, the user
can replace the new model with the old one to be used for predictions.
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import train_model
from dash import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, log_loss


# Create Dash app
dash.register_page(__name__, path='/train-model')

# Sidebar layout with dbc components
sidebar = dbc.Col(
    [
        html.H2("Model Metrics", className='display-6'), #  title
        html.Hr(),
        html.P("Current Model Performance Metrics:", className="lead"), # header        
        html.Div(id='accuracy-display', style={'margin-bottom': '10px'}), # accuracy of current model
        html.Div(id='loss-display', style={'margin-bottom': '10px'}), # loss of current model

        html.Hr(),
        html.P("New Model Performance Metrics:", className="lead"), # header
        html.Div(id='new-accuracy-display', style={'margin-bottom': '10px'}), # accuracy of new model
        html.Div(id='new-loss-display', style={'margin-bottom': '10px'}), # loss of new model

        dbc.Button("Train New Model", id='train-button', color='primary', className='mt-3'), # button to train a new model
        html.Div(id='training-message', className='mt-3'), # message for when training is completed
        dbc.Button("Replace Current Model", id='replace-button', color='success', className='mt-3', disabled=True), # button to replace with new model
        html.Div(id='replace-message', className='mt-3') # message for successful replacement
    ],
    width=2,
    style={
        'padding': '20px',
        'background-color': '#f8f9fa',
        'height': '100vh',
        'position': 'fixed'
    }   
)

# Content layout to show the transactions DataTable
content = dbc.Col(
    [
        html.H1("Model Evaluation", className='mt-4 mb-4'),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='confusion-matrix'), width=6), # confusion matrix of current model
                dbc.Col(dcc.Graph(id='new-confusion-matrix'), width=6) # confusion matrix of new model
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='current-true-label-distribution'), width=6), # distribution of classes (true)
                dbc.Col(dcc.Graph(id='current-pred-label-distribution'), width=6) # distribution of classes (predicted)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='new-true-label-distribution'), width=6), # distribution of classes (true)
                dbc.Col(dcc.Graph(id='new-pred-label-distribution'), width=6) # distribution of classes (predicted)
            ]
        )
    ],
    width=9,
    style={
        'margin-left': '320px',
        'padding': '20px'
    }
)

# Main layout with the sidebar and the main content
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
    [
        Output('accuracy-display', 'children'),
        Output('loss-display', 'children'),
        Output('confusion-matrix', 'figure'),
        Output('current-true-label-distribution', 'figure'),
        Output('current-pred-label-distribution', 'figure')
    ],
    [
        Input('url', 'pathname'), 
        Input('transactions-store', 'data')
    ]
)
def update_model_metrics(_, table_data):
    """The callback function to update the accuracy, loss, confusion matrix, and distribution plots
    for the current model. The function updates with every page refresh for whenever new data is updated
    in the transaction database.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - _: for page refreshes
    - table_data: the transactions store that was extracted in app.py that is all the transaction data
    from the transactions database

    Output:
    - accuracy-display: a html display of the current accuracy of the model
    - loss-display: a html display of the current loss of the model
    - confusion-matrix: a plotly heatmap of the confusion matrix of the model
    - current-true-label-distribution: a distribution of the true classes that I labelled (taken from
    the 'labels' column in the transactions database) 
    """

    # extract data from the transactions store
    df = pd.DataFrame(table_data)

    # ensure that all the data is labelled and has predictions for each transaction
    df = df[(~df['labels'].isna()) | (~df['predicted_labels'].isna())]

    if df.empty:
        return "Accuracy: No data available", "Loss: No data available", go.Figure(), go.Figure()
    
    # define the true labels and the predicted labels
    y_true = df['labels']
    y_pred = df['predicted_labels']

    # define the classes and number them by the index
    classes = y_true.unique()
    label_mapping = {label: idx for idx, label in enumerate(classes)}

    # map the values of the indices to the class names (used for accuracy and loss calculations)
    y_true_num = y_true.map(label_mapping)
    y_pred_num = y_pred.map(label_mapping)

    # one hot encode the labels
    y_true_onehot = pd.get_dummies(y_true_num, columns=classes)
    y_pred_onehot = pd.get_dummies(y_pred_num, columns=classes).reindex(columns=y_true_onehot.columns, fill_value=0)

    # calculate the accuracy score and the log loss
    accuracy = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true_onehot, y_pred_onehot)

    # define colors based on the accuracy
    if accuracy > 0.9:
        accuracy_color = 'green'
    elif 0.8 <= accuracy <= 0.9:
        accuracy_color = 'yellow'
    else:
        accuracy_color='red'

    # create the confusion matrix
    cm = confusion_matrix(y_true_num, y_pred_num)

    # display the confusion matrix through a heatmap
    cm_fig = px.imshow(
        cm,
        labels=dict(x='Predicted Label', y="True Label", color="Count"),
        x=classes,
        y=classes,
        title="Current Confusion Matrix",
        text_auto=True
    )
    cm_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=800,
        height=800,
        font=dict(size=14),
        coloraxis_showscale=False
    )

    # display accuracy and loss
    accuracy_display = html.Div(
        children = [
            html.Span('Accuracy: ', style={'font-weight': 'bold'}),
            html.Span(f'{accuracy:.2f}', style={"background-color": accuracy_color, "padding": "5px", "border-radius": "5px"})
        ]
    )

    loss_display = html.Div(
        children=[
            html.Span("Loss: ", style={"font-weight": "bold"}),
            html.Span(f"{loss:.2f}", style={"background-color": "#f0f0f0", "padding": "5px", "border-radius": "5px"})
        ]
    )

    # plots for distributions for the true and predicted labels
    true_label_counts = y_true.value_counts().sort_values(ascending=False)
    true_label_fig = px.bar(
        x=true_label_counts.index,
        y=true_label_counts.values,
        title='Current True Category Distribution',
        labels={'x':'Category', 'y':'Count'},
        color_discrete_sequence=['#008080'] 
    )

    true_label_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=700,
        height=500,
        font=dict(size=14)
    )

    predicted_label_counts = y_pred.value_counts().sort_values(ascending=False)
    predicted_label_fig = px.bar(
        x=predicted_label_counts.index,
        y=predicted_label_counts.values,
        title='Current Predicted Category Distribution',
        labels={'x': 'Category', 'y': 'Count'},
        color_discrete_sequence=['#FF6F61']
    )
    predicted_label_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=700,
        height=500,
        font=dict(size=14)
    )

    return accuracy_display, loss_display, cm_fig, true_label_fig, predicted_label_fig

# load the tokenizer and the embed model from train_model.py (using DistilBERT model)
tokenizer, embed_model = train_model.load_embed_model()

@callback(
    [
        Output('training-message', 'children'),
        Output('replace-button', 'disabled'),
        Output('new-confusion-matrix', 'figure'),
        Output('new-true-label-distribution', 'figure'),
        Output('new-pred-label-distribution', 'figure'),
        Output('new-accuracy-display', 'children'),
        Output('new-loss-display', 'children')
    ],
    [Input('train-button', 'n_clicks')],
    [State('transactions-store', 'data')]
)
def train_new_model(n_clicks, table_data):
    """The callback function to update the accuracy, loss, confusion matrix, and distribution plots
    for a newly trained model. The function updates every time the user clicks on the model.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - n_clicks: Calls the function and begins training a new model
    - table_data: the transactions store that was extracted in app.py that is all the transaction data
    from the transactions database

    Output:
    - training-message: a message to let the use know whne the model is finished training
    - replace-button: enables the button to allow the user to replace the current model with the new model
    if they are satisfied with the results
    - new-confusion-matrix: a plotly heatmap of the confusion matrix of the new model
    - new-true-label-distribution: a distribution of the true classes that I labelled (taken from
    the 'labels' column in the transactions database) 
    - new-pred-label-distribution: a distribution of the predicted classes from the new model
    - new-accuracy-display: a html display of the new accuracy of the model
    - new-loss-display: a html display of the new loss of the model
    """

    # used for when the user does not click on the button
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # extract the data from the transactions database into a DataFrame
    df = pd.DataFrame(table_data)

    # ensure that the labels and predicted labels is filled out
    df = df[(~df['labels'].isna()) & (~df['predicted_labels'].isna())]

    # process the data so it will work in the model
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(str).str.replace(',', '').astype(float)

    # drop predicted_labels column since we will be creating new predictions
    df = df.drop(columns='predicted_labels')

    # extract_and_preprocess_txns will split the data in train, validation, and test datasets as well as give the label binarizer
    # the label binarizer will be called label_encoder
    label_encoder, X_y_dict = train_model.extract_and_preprocess_txns(df=df, db_filepath=None, train_or_predict='train')
    X_train = X_y_dict['X_train']
    y_train = X_y_dict['y_train']
    X_val = X_y_dict['X_val']
    y_val = X_y_dict['y_val']
    X_test = X_y_dict['X_test']
    y_test = X_y_dict['y_test']

    # load the pipeline used for preprocessing the data
    print('Loading pipeline...')
    pipeline = joblib.load('preprocess_pipeline_fitted.pkl')

    # fit a new pipeline with the X_train data 
    pipeline.fit(X_train)

    # save the new pipeline temporarily
    joblib.dump(pipeline, 'temp_preprocess_pipeline_fitted.pkl')
    print('New Pipeline fitted and saved!')

    # transform the X datasets using the newly fitted pipeline
    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    print('Datasets processed. Embedding text now...')

    # embed the text data of the transactions for each dataset using the preloaded tokenizer and model
    X_train_embeddings = train_model.embed_text(X_train_processed['extended_text'], tokenizer, embed_model)
    X_val_embeddings = train_model.embed_text(X_val_processed['extended_text'], tokenizer, embed_model)
    X_test_embeddings = train_model.embed_text(X_test_processed['extended_text'], tokenizer, embed_model)

    print('Text embedded')

    # split the data into two arrays: one array with all features except the embedded text and another for the embedded text.
    # also remove the id column and keep that just in case (X_train_id)
    X_train_other, X_train_embeddings, X_train_id = train_model.embedding_otherfeatures_split(X_train_processed, X_train_embeddings)
    X_val_other, X_val_embeddings, X_val_id = train_model.embedding_otherfeatures_split(X_val_processed, X_val_embeddings)
    X_test_other, X_test_embeddings, X_test_id = train_model.embedding_otherfeatures_split(X_test_processed, X_test_embeddings)

    # build the new model (this model is an ensemble model with logisitic regression, neural network, and SVC)
    model = train_model.CustomModel(embeddings_shape=X_train_embeddings.shape[1],
                                other_features_shape=X_train_other.shape[1],
                                num_categories=y_train.shape[1],
                                learning_rate=1e-5)

    # fit the new model
    model.fit(X_train_embeddings, X_train_other, X_val_embeddings, X_val_other, y_train, y_val, 5, 2)

    # temporarily save the new model in the temp_models folder
    model.save('./temp_models')

    # calculate accuracies
    train_accuracy  = model.evaluate(X_train_embeddings, X_train_other, y_train)
    val_accuracy = model.evaluate(X_val_embeddings, X_val_other, y_val)
    test_accuracy = model.evaluate(X_test_embeddings, X_test_other, y_test)

    # get the class labels from label_encoder
    classes = label_encoder.classes_
    class_indices = np.arange(len(classes)).reshape(-1, 1)

    # predict test classes
    y_test_pred = model.predict(X_test_embeddings, X_test_other)

    # OneHotEncode the predictions
    onehot_encoder = OneHotEncoder(categories=[class_indices.flatten()],sparse=False)
    onehot_encoder.fit(class_indices)
    y_test_pred = y_test_pred.reshape(-1,1)
    y_test_pred = onehot_encoder.transform(y_test_pred)

    test_loss = log_loss(y_test, y_test_pred)

    # determine color for accuracy display
    if test_accuracy > 0.9:
        accuracy_color = 'green'
    elif 0.8 <= test_accuracy <= 0.9:
        accuracy_color = 'yellow'
    else:
        accuracy_color = 'red'

    # update accuracy and loss displays for the new model
    new_accuracy_display = html.Div(
        children=[
            html.Span('New Test Accuracy: ', style={'font-weight': 'bold'}),
            html.Span(f'{test_accuracy:.2f}', style={"background-color": accuracy_color, "padding": "5px", "border-radius": "5px"})
        ]
    )

    new_loss_display = html.Div(
        children=[
            html.Span("New Test Loss: ", style={"font-weight": "bold"}),
            html.Span(f"{test_loss:.2f}", style={"background-color": "#f0f0f0", "padding": "5px", "border-radius": "5px"})
        ]
    )

    # convert y_test and y_test_pred to their label encoded versions
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_test_pred, axis=1)

    # calculate and display the confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=np.arange(len(classes)))
    cm_fig = px.imshow(
        cm,
        labels=dict(x='Predicted Label', y='True Label', color='Count'),
        x=classes,
        y=classes,
        title='New Model Confusion Matrix',
        text_auto=True
    )
    cm_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=800,
        height=800,
        font=dict(size=14),
        coloraxis_showscale=False
    )

    # plot the true label distributions
    true_label_counts = pd.Series(y_true_labels).value_counts().sort_values(ascending=False)
    true_label_counts.index = true_label_counts.index.map(lambda idx: classes[idx])
    true_label_fig = px.bar(
        x=true_label_counts.index,
        y=true_label_counts.values,
        title='New Model True Category Distribution',
        labels={'x': 'Category', 'y': 'Count'},
        color_discrete_sequence=['#008080']
    )
    true_label_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=700,
        height=500,
        font=dict(size=14)
    )

    # plot the predicted label distributions
    predicted_label_counts = pd.Series(y_pred_labels).value_counts().sort_values(ascending=False)
    predicted_label_counts.index = predicted_label_counts.index.map(lambda idx: classes[idx])
    predicted_label_fig = px.bar(
        x=predicted_label_counts.index,
        y=predicted_label_counts.values,
        title='New Model Predicted Category Distribution',
        labels={'x': 'Category', 'y': 'Count'},
        color_discrete_sequence=['#FF6F61']
    )
    predicted_label_fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        width=700,
        height=500,
        font=dict(size=14)
    )

    return (
        "Training successful! You may now replace the current model with this new model.",
        False,  # Enable the replace button
        cm_fig,
        true_label_fig,
        predicted_label_fig,
        new_accuracy_display,
        new_loss_display
    )

@callback(
    [
        Output('replace-message', 'children'),
        Output('replace-button', 'disabled', allow_duplicate=True)
    ],
    [Input('replace-button', 'n_clicks')],
    prevent_initial_call=True
)
def replace_model(n_clicks):
    """The callback function to replace the current model with the newly trained model. If successful,
    the button will once again be disabled until the next time a model is trained.
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - n_clicks: Calls the function and begins replacing the model

    Outputs:
    - replace-message: Notifies the user whether the model was successfully replaced or not
    - replace-button: The button will be disabled once the model is successfully replaced
    """

    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    try:
        # define paths
        new_model_dir = './temp_models'  # temporary location for the new model
        current_model_dir = './current_models'  # location where the current model is saved

        # load the new model from the temporary directory
        new_model = CustomModel.load(new_model_dir)

        # replace the current model
        # save the loaded new model in the current model directory
        if not os.path.exists(current_model_dir):
            os.makedirs(current_model_dir)

        # saving the new model over the old one
        new_model.save(current_model_dir)

        # provide a success message
        replace_message = "The current model has been successfully replaced with the new model."

        return replace_message, True  # disable the replace button after successful replacement

    except Exception as e:
        # in case of an error, provide an error message
        replace_message = f"Error occurred while replacing the model: {str(e)}"
        return replace_message, False  # keep the button enabled if replacement fails