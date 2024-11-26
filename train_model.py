import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

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

import joblib


def extract_and_preprocess_txns(df, db_filepath=None, train_or_predict='train'):
    """Get the transaction data (either through a given dataframe or through a database filepath) and
    split into dataframes of features (X) and target (y). Then split into train, validation, and test
    datasets if the train_or_predict is set to train. If set to predict, only provide the X data
    ----------------------------------------------------------------------------------------------------
    Inputs:
    - df: DataFrame of transactions, if provided.
    - db_filepath: will extract transactions from the database if a filepath is provided.

    Outputs:
    - label_encoder: a LabelBinarizer object that can be used to extract the class names for the target
    variable
    - X or X_y_dict: X is used for prediction, it is a DataFrame of all transaction data without the labels.
    X_y_dict is a dictionary of DataFrames that is the train, validation, and test sets.
    """

    if db_filepath is not None:
        # if a filepath is provided to a database, connect to database
        # query all transaction data where the labels is not missing
        conn = sqlite3.connect(db_filepath)
        extract_query = 'SELECT * FROM transactions WHERE labels IS NOT NULL'
        txns_df = pd.read_sql(extract_query, conn)
    else:
        # else use the provided dataframe
        txns_df = df
    
    # get X (or the features for transactions)
    X = txns_df.drop(columns=['labels'])

    if train_or_predict == 'train':
        # if train is selected create a dictionary of train, validation, and test data

        # encode the labels column
        labels = txns_df['labels']
        label_encoder = LabelBinarizer()
        y = label_encoder.fit_transform(labels)

        # split the X and y data into 60/20/20 split of train, validation, and test data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

        # create dictionary of all the datasets
        X_y_dict = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        return label_encoder, X_y_dict
    elif train_or_predict == 'predict':
        # if set to predict, just return the X data
        return None, X
    else:
        # error just for incorred input
        print("Incorrect Input: Accept only 'train' or 'predict'")

def load_embed_model(model_id = "distilbert-base-uncased-finetuned-sst-2-english"):
    """This function is used just for loading a tokenizer and pretrained embedding model.
    Default is a variation of distilbert"""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = TFAutoModel.from_pretrained(model_id)

    return tokenizer, model

def embed_text(text_series, tokenizer, model):
    """This function is used to tokenize and embed the text data from transactions"""

    # tokenize the text_series (text feature in transactions)
    inputs = tokenizer(text_series.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=128)

    # embed the tokenized text
    outputs = model(inputs)
    embeddings = outputs.last_hidden_state
    embeddings = tf.reduce_mean(embeddings, axis=1) # global avg pooling
    return embeddings

def embedding_otherfeatures_split(X_df, X_embedding_df):
    "Function for processing the X variable and X embeddings"

    # get the ids separately as it is not used for prediction in the model
    transaction_ids = X_df['transaction_id']  

    # drop the text column and id columns as they are stored elsewhere
    # also convert to numpy arrays
    X_df = X_df.drop(columns=['extended_text', 'transaction_id']).to_numpy()
    X_embedding_df = X_embedding_df.numpy()

    return X_df, X_embedding_df, transaction_ids

def store_predictions(db_filepath, id_series, predicted_labels):
    """
    Store the predicted labels in the transactions.db database.
    
    If the 'predicted_labels' column does not exist, create it. 
    If it exists, update the existing column.

    Parameters:
    ------------------------------------------
    db_filepath (str): Path to the SQLite database file.
    df_with_predictions (DataFrame): DataFrame that contains the transaction_id and predicted labels.
        It must have columns ['transaction_id', 'predicted_labels'].
    """
    df_with_predictions = pd.DataFrame({
        'transaction_id': id_series,
        'predicted_labels': predicted_labels
    })

    # Connect to the SQLite DB
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()

    # Check if 'predicted_labels' column already exists
    try:
        cursor.execute('SELECT predicted_labels FROM transactions LIMIT 1')
    except sqlite3.OperationalError:
        # If the column doesn't exist, add it
        add_column_query = "ALTER TABLE transactions ADD COLUMN predicted_labels TEXT"
        cursor.execute(add_column_query)
        conn.commit()

    # Prepare the update query
    update_query = '''
    UPDATE transactions
    SET predicted_labels = ?
    WHERE transaction_id = ?
    '''
    
    # Convert DataFrame to a list of tuples (predicted_label, transaction_id)
    data_to_update = df_with_predictions[['predicted_labels', 'transaction_id']].values.tolist()

    # Update the transactions table with new predictions
    cursor.executemany(update_query, data_to_update)
    conn.commit()

    # Close the connection
    conn.close()

    print("Predicted labels successfully stored in the database.")

class CustomModel:
    """This CustomModel class is used to build a neural network softmax classifier, logistic regression model,
    and support vector classifier. It utilizes softvoting to choose the best classifier. The class also has the 
    ability to load and save into a designated path.
    
    Attributes
    --------------
    embeddings_shape: shape of the embeddings array
    other_features_shape: shape of the array that contains all other features
    num_categories: number of classes
    learning_rate: learning rate of the model
    nn_model: the neural network classifier
    logistic_model: logistic regression model
    svc_model: support vector classifier

    Methods
    -------------
    build_neural_network(self, embeddings_shape, other_features_shape, num_categories, learning_rate):
        builds the neural network with a set architecture. The neural network will process the embedding array and 
        other_features array separately and concatenate them to make the final softmax output.
    
    fit(self, X_train_embeddings, X_train_other, X_val_embeddings, X_val_other, y_train, y_val, num_epochs, batch_size):
        fits all three models (nn_model, logistic_model, and svc_model)

    predict(self, X_embeddings, X_other):
        uses soft voting and combines the probability distributions of all three models to choose the most probable class
    
    evaluate(self, X_embeddings, X_other, y_true):
        calculates the accuracy of the model
    
    save(self, path):
        saves all three models into their own set path
    
    load(cls, path):
        load all three models from a set path
    """
    def __init__(self, embeddings_shape=None, other_features_shape=None, num_categories=None, learning_rate=0.001):
        if embeddings_shape is not None and other_features_shape is not None and num_categories is not None:
            # initialize TensorFlow model
            self.nn_model = self.build_neural_network(embeddings_shape, other_features_shape, num_categories, learning_rate)
        
        # initialize Logistic Regression and SVC models
        self.logistic_model = LogisticRegression(max_iter=5000, random_state=42)
        self.svc_model = SVC(kernel='linear', C=0.09, probability=True, random_state=42)
        self.num_categories = num_categories
    
    def build_neural_network(self, embeddings_shape, other_features_shape, num_categories, learning_rate):
        """build a Neural Network that takes embeddings and other features as input."""

        embeddings_input = tf.keras.layers.Input(shape=(embeddings_shape,))
        other_features_input = tf.keras.layers.Input(shape=(other_features_shape,))

        # reduce the size of the embeddings tensor to 150 neurons
        reduction_layer = tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(embeddings_input)

        # process other features
        other_features_layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(other_features_input)
        other_features_layer = tf.keras.layers.Dropout(0.1)(other_features_layer)

        # concatenate embedding and other features
        concat_layer = tf.keras.layers.concatenate([reduction_layer, other_features_layer])

        # dense layer with dropout
        dense_layer = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat_layer)
        dropout_layer = tf.keras.layers.Dropout(0.1)(dense_layer)

        # output layer with softmax
        output_layer = tf.keras.layers.Dense(num_categories, activation='softmax')(dropout_layer)

        model = tf.keras.Model(inputs=[embeddings_input, other_features_input], outputs=output_layer)

        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def _plot_loss(self, history):
        """Helper function to plot loss vs epochs after training from the history object"""
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = np.arange(len(train_loss))

        fig, ax = plt.subplots()
        ax.plot(epochs, train_loss, marker='o', linestyle='-', label='Training Loss')
        ax.plot(epochs, val_loss, marker='o', linestyle='-', label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Neural Network Loss vs Epoch Plot')
        ax.legend()
        plt.show()

    def fit(self, X_train_embeddings, X_train_other, X_val_embeddings, X_val_other, y_train, y_val, num_epochs, batch_size):
        """Fit the neural network and scikit-learn models, and plot the loss vs epochs."""

        # create callbacks to stop and save the model based on validation loss
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)

        # fit the TensorFlow model
        history = self.nn_model.fit(
            x=[X_train_embeddings, X_train_other],
            y=y_train,
            validation_data=([X_val_embeddings, X_val_other], y_val),
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # train scikit-learn models
        X_train_concat = np.concatenate([X_train_embeddings, X_train_other], axis=1)
        self.logistic_model.fit(X_train_concat, y_train.argmax(axis=1))
        self.svc_model.fit(X_train_concat, y_train.argmax(axis=1))

        # set classes_ attribute for both models to ensure that all models have the same # of classes
        all_classes = np.arange(self.num_categories)
        self.logistic_model.classes_ = all_classes
        self.svc_model.classes_ = all_classes

    def predict(self, X_embeddings, X_other):
        """Predict the outcomes using a soft voting approach."""

        # predict with TensorFlow Model
        nn_predictions = self.nn_model.predict([X_embeddings, X_other])
        # combine embeddings and other features for scikit-learn models
        X_concat = np.concatenate([X_embeddings, X_other], axis=1)

        # predict probabilities with Logistic Regression and SVC
        logistic_proba = self.logistic_model.predict_proba(X_concat)
        svc_proba = self.svc_model.predict_proba(X_concat)

        # soft voting: Average the probabilities
        combined_proba = (nn_predictions + logistic_proba + svc_proba) / 3

        # final prediction based on highest averaged probability
        final_predictions = np.argmax(combined_proba, axis=1)

        return final_predictions

    def evaluate(self, X_embeddings, X_other, y_true):
        """Evaluate the model's performance. Return the accuracy"""
        y_pred = self.predict(X_embeddings, X_other)
        accuracy = np.mean(y_pred == y_true.argmax(axis=1))

        return accuracy

    def save(self, path):
        """Save the entire ensemble model."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # save TensorFlow model
        tf_model_path = os.path.join(path, 'nn_model')
        self.nn_model.save(tf_model_path)

        # save Logistic Regression model
        logistic_model_path = os.path.join(path, 'logistic_model.pkl')
        joblib.dump(self.logistic_model, logistic_model_path)

        # save SVC model
        svc_model_path = os.path.join(path, 'svc_model.pkl')
        joblib.dump(self.svc_model, svc_model_path)

    @classmethod
    def load(cls, path):
        """Load the entire ensemble model."""
        # load TensorFlow model
        tf_model_path = os.path.join(path, 'nn_model')
        nn_model = tf.keras.models.load_model(tf_model_path)

        # load Logistic Regression model
        logistic_model_path = os.path.join(path, 'logistic_model.pkl')
        logistic_model = joblib.load(logistic_model_path)

        # load SVC model
        svc_model_path = os.path.join(path, 'svc_model.pkl')
        svc_model = joblib.load(svc_model_path)

        # create an instance of CustomModel
        instance = cls()

        # assign loaded models to the instance
        instance.nn_model = nn_model
        instance.logistic_model = logistic_model
        instance.svc_model = svc_model

        return instance