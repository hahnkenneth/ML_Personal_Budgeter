"""
This file is used to preprocess the transaction data. It is used to create a pipeline
that will enable the transaction data to be useable by a model. The pipeline will be
saved as preprocess_pipeline.pkl once this module is run.

RUN THIS FILE SECOND before running the app.
"""

import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    RobustScaler,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocesses the text column in the data to remove excess whitespace, punctuations, and numbers"""
    def __init__(self, text_columns):
        self.text_columns = text_columns
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        X_copy = X.copy()
        for col in self.text_columns:
            X_copy[col] = X_copy[col].fillna('').astype(str)
            # some initial preprocessing then apply regex substitutions
            X_copy[col] = X_copy[col].str.lower().str.replace('\n', ' ', regex=False).apply(self._preprocess_text)
        X_copy['extended_text'] = X_copy[self.text_columns].agg(' '.join, axis=1)
        X_copy = X_copy.drop(columns=self.text_columns)
        return X_copy
    def _preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numeric values
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        # Remove repeated words
        words = text.split()
        unique_words = list(dict.fromkeys(words))
        return ' '.join(unique_words)

class DateFeatureProcessor(BaseEstimator, TransformerMixin):
    """Extract useful information from the date column"""
    def __init__(self, date_col):
        self.date_col = date_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col])
        X_copy['dow'] = X_copy[self.date_col].dt.dayofweek
        X_copy['dom'] = X_copy[self.date_col].dt.day
        X_copy['month'] = X_copy[self.date_col].dt.month
        X_copy['year'] = X_copy[self.date_col].dt.year
        X_copy.drop(columns=[self.date_col], inplace=True)
        return X_copy
    
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Cyclically encode the day of week and the month columns in order for model to learn about patterns"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'dow' in self.columns:
            X_copy['dow_sin'] = np.sin(2 * np.pi * X_copy['dow'] / 7)
            X_copy['dow_cos'] = np.cos(2 * np.pi * X_copy['dow'] / 7)
        if 'month' in self.columns:
            X_copy['month_sin'] = np.sin(2 * np.pi * X_copy['month'] / 12)
            X_copy['month_cos'] = np.cos(2 * np.pi * X_copy['month'] / 12)
        X_copy.drop(columns=self.columns, inplace=True)
        return X_copy


class ColumnScaler(BaseEstimator, TransformerMixin):
    """Custom transformer that standard or robust scales a specific column."""
    def __init__(self, column):
        self.column = column
        if column == 'amount':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Convert column values to numeric before fitting the scaler
        X = self._process_numeric(X)
        # Fit the scaler on the specific column (convert to numpy array to fit)
        self.scaler.fit(X[[self.column]].values)
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Convert column values to numeric before transforming
        X_copy = self._process_numeric(X_copy)
        # Transform the specific column (convert to numpy array to transform)
        X_copy[self.column] = self.scaler.transform(X_copy[[self.column]].values)
        
        return X_copy
    
    def _process_numeric(self, X):
        """Remove commas and convert to float for a specific column."""
        X_copy = X.copy()
        # Remove commas and convert to float
        X_copy[self.column] = (
            X_copy[self.column]
            .astype(str)  # Ensure it is treated as string
            .str.replace(',', '')  # Remove commas
            .astype(float)  # Convert to float
        )
        return X_copy

class MultiColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer that one-hot encodes multiple columns and adds them back to the DataFrame."""
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {col: OneHotEncoder(sparse=False) for col in columns}  # Create a separate encoder for each column
    def fit(self, X, y=None):
        # Fit the encoder on each specific column
        for col in self.columns:
            self.encoders[col].fit(X[[col]])  # Fit each encoder to its respective column
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_frames = []

        # Transform each column and store the encoded columns
        for col in self.columns:
            encoded_cols = self.encoders[col].transform(X_copy[[col]])
            encoded_df = pd.DataFrame(
                encoded_cols, 
                columns=[f"{col}_{category}" for category in self.encoders[col].categories_[0]],
                index=X_copy.index
            )
            encoded_frames.append(encoded_df)
            # Drop the original column
            X_copy.drop(columns=[col], inplace=True)

        # Concatenate all encoded columns with the remaining original DataFrame
        X_copy = pd.concat([X_copy] + encoded_frames, axis=1)

        return X_copy

# Columns to preprocess
text_columns = ['description', 'extended_details', 'merchant', 'address', 'city', 'state', 'zip_code', 'category']
date_column = 'date'
cyclical_columns = ['dow', 'month']
onehot_columns = ['year', 'source']
standard_scale_column = 'dom'
robust_scale_column = 'amount'

preprocess_pipeline = Pipeline([
    ('date_feature_extractor', DateFeatureProcessor(date_col=date_column)),
    ('cyclical_encoder', CyclicalEncoder(columns=cyclical_columns)),
    ('text_processor', TextPreprocessor(text_columns=text_columns)),
    ('one_hot_encoder', MultiColumnOneHotEncoder(onehot_columns)),
    ('standard_scaler', ColumnScaler(standard_scale_column)),
    ('robust_scaler', ColumnScaler(robust_scale_column))
])

joblib.dump(preprocess_pipeline, 'preprocess_pipeline.pkl')