from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def split_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and validation sets.

    :param df: DataFrame containing the data to be split
    :param test_size: Proportion of the dataset to include in the validation set
    :param random_state: Random state seed for reproducibility
    :return: Tuple containing the training and validation DataFrames
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, val_df


def extract_columns(df: pd.DataFrame, input_columns: list, target_column: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract input features and target columns from the dataset.

    :param df: DataFrame from which to extract columns
    :param input_columns: List of column names to use as input features
    :param target_column: List of column names to use as the target
    :return: Tuple containing the input features and target DataFrames
    """
    inputs = df[input_columns]
    targets = df[target_column]
    return inputs, targets


def get_column_types(df: pd.DataFrame, input_columns: list) -> Tuple[list, list]:
    """
    Identify numeric and categorical columns in the dataset.

    :param df: DataFrame containing the data
    :param input_columns: List of input feature column names
    :return: Tuple containing lists of numeric and categorical column names
    """
    numeric_cols = df[input_columns].select_dtypes(exclude='object').columns.tolist()
    categorical_cols = df[input_columns].select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols


def encode_categorical_data(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply one-hot encoding to categorical columns in the dataset.

    :param train_inputs: Training input features DataFrame
    :param val_inputs: Validation input features DataFrame
    :param categorical_cols: List of categorical column names
    :return: Tuple containing transformed training and validation DataFrames
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    # Apply the transformation
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

    return train_inputs, val_inputs


def scale_numeric_data(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply MinMax scaling to numeric columns in the dataset.

    :param train_inputs: Training input features DataFrame
    :param val_inputs: Validation input features DataFrame
    :param numeric_cols: List of numeric column names
    :return: Tuple containing scaled training and validation DataFrames
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])

    # Apply the transformation
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    return train_inputs, val_inputs


def preprocess_data(df: pd.DataFrame, scale_numeric: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the dataset by splitting, encoding categorical data, and scaling numeric data.
    The input columns and target column are predefined.

    :param df: DataFrame containing the dataset
    :return: Tuple containing processed training inputs, training targets, validation inputs, and validation targets
    """
    # Predefined input and target columns
    input_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    target_column = ['Exited']

    # Split the data
    train_df, val_df = split_data(df)

    # Extract input and target columns
    train_inputs, train_targets = extract_columns(train_df, input_columns, target_column)
    val_inputs, val_targets = extract_columns(val_df, input_columns, target_column)

    # Get numeric and categorical columns
    numeric_cols, categorical_cols = get_column_types(df, input_columns)

    # Encode categorical data
    train_inputs, val_inputs = encode_categorical_data(train_inputs, val_inputs, categorical_cols)

    # Scale numeric data
    if scale_numeric:
        train_inputs, val_inputs = scale_numeric_data(train_inputs, val_inputs, numeric_cols)

    return train_inputs, train_targets, val_inputs, val_targets

    
def preprocess_new_data(new_data: pd.DataFrame, encoder: OneHotEncoder, scaler: MinMaxScaler = None) -> pd.DataFrame:
    """
    Preprocess new data using previously trained OneHotEncoder and MinMaxScaler.

    :param new_data: DataFrame containing the new data to be processed
    :param encoder: Previously trained OneHotEncoder for categorical data transformation
    :param scaler: Previously trained MinMaxScaler for numeric data scaling (optional)
    :return: Preprocessed new data
    """
    # Predefined input columns
    input_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    # Get numeric and categorical columns
    numeric_cols = new_data[input_columns].select_dtypes(exclude='object').columns.tolist()
    categorical_cols = new_data[input_columns].select_dtypes(include='object').columns.tolist()

    # Transform categorical data
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_data[encoded_cols] = encoder.transform(new_data[categorical_cols])

    # Optionally scale numeric data
    if scaler:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    return new_data