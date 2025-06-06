import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import Tuple, List, Optional, Dict, Any

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and validation sets.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets.
    """
    return train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)

def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        Tuple[List[str], List[str]]: Lists of numeric and categorical column names.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes('object').columns.tolist()
    
    # Remove specific columns
    for col in ['CustomerId', 'id']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if 'Surname' in categorical_cols:
        categorical_cols.remove('Surname')
    
    return numeric_cols, categorical_cols

def scale_numeric_data(train_data: pd.DataFrame, val_data: pd.DataFrame, numeric_cols: List[str], scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Scale numeric data using either StandardScaler or MinMaxScaler.

    Args:
        train_data (pd.DataFrame): Training data.
        val_data (pd.DataFrame): Validation data.
        numeric_cols (List[str]): List of numeric column names.
        scaler_type (str): Type of scaler to use ('standard' or 'minmax').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, object]: Scaled training data, scaled validation data, and the fitted scaler.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaler.fit(train_data[numeric_cols])
    train_data[numeric_cols] = scaler.transform(train_data[numeric_cols])
    val_data[numeric_cols] = scaler.transform(val_data[numeric_cols])
    
    return train_data, val_data, scaler

def encode_categorical_data(train_data: pd.DataFrame, val_data: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Encode categorical data using OneHotEncoder.

    Args:
        train_data (pd.DataFrame): Training data.
        val_data (pd.DataFrame): Validation data.
        categorical_cols (List[str]): List of categorical column names.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]: Encoded training data, encoded validation data, 
        the fitted encoder, and the list of encoded column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')
    encoder.fit(train_data[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_data[encoded_cols] = encoder.transform(train_data[categorical_cols])
    val_data[encoded_cols] = encoder.transform(val_data[categorical_cols])
    
    return train_data, val_data, encoder, encoded_cols

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features (X) and target (y).

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def preprocess_data(df: pd.DataFrame, scaler_numeric: bool = True, scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object, OneHotEncoder]:
    """
    Preprocess the data by splitting, scaling numeric features, and encoding categorical features.

    Args:
        df (pd.DataFrame): The input dataframe.
        scaler_numeric (bool): Whether to scale numeric features.
        scaler_type (str): Type of scaler to use if scaling ('standard' or 'minmax' or 'None').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object, OneHotEncoder]: 
        Preprocessed training features, validation features, training targets, validation targets, 
        fitted scaler (or None if not used), and fitted encoder.
    """

    target_col = 'Exited'
    train_set, val_set = split_data(df, target_col)
    
    train_inputs, y_train = split_features_target(train_set, target_col)
    val_inputs, y_val = split_features_target(val_set, target_col)
    
    numeric_cols, categorical_cols = get_column_types(train_inputs )
    
    scaler = None
    if scaler_numeric:
        train_inputs , val_inputs , scaler = scale_numeric_data(train_inputs , val_inputs , numeric_cols, scaler_type)
    
    train_inputs , val_inputs , encoder, encoded_cols = encode_categorical_data(train_inputs , val_inputs , categorical_cols)
    
    final_cols = numeric_cols + encoded_cols
    X_train = train_inputs[final_cols]
    X_val = val_inputs[final_cols]
    
    return {
        'train_X': X_train,
        'train_y': y_train,
        'val_X': X_val,
        'val_y': y_val,
        'scaler': scaler,
        'encoder': encoder,
        'encoded_cols': encoded_cols,
        'numeric_cols': numeric_cols
    }
    
# X_train, X_val, y_train, y_val, scaler, encoder

# def preprocess_new_data(new_data: pd.DataFrame, scaler: Optional[object], encoder: OneHotEncoder, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
#     """
#     Preprocess new data using the fitted scaler and encoder.

#     Args:
#         new_data (pd.DataFrame): New data to preprocess.
#         scaler (Optional[object]): Fitted scaler (StandardScaler or MinMaxScaler) or None.
#         encoder (OneHotEncoder): Fitted OneHotEncoder.
#         numeric_cols (List[str]): List of numeric column names.
#         categorical_cols (List[str]): List of categorical column names.

#     Returns:
#         pd.DataFrame: Preprocessed new data.
#     """
#     numeric_cols, categorical_cols = get_column_types(new_data)

#     if scaler is not None:
#         new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
#     encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#     new_data[encoded_cols] = encoder.transform(new_data[categorical_cols])
    
#     return new_data[numeric_cols + encoded_cols]

def preprocess_new_data(new_data: pd.DataFrame, preprocessing_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess new data using the fitted scaler, encoder, and column information.

    Args:
        new_data (pd.DataFrame): New data to preprocess.
        preprocessing_info (Dict[str, Any]): Dictionary containing preprocessing objects and column information.

    Returns:
        pd.DataFrame: Preprocessed new data.
    """
    scaler = preprocessing_info['scaler']
    encoder = preprocessing_info['encoder']
    numeric_cols = preprocessing_info['numeric_cols']
    encoded_cols = preprocessing_info['encoded_cols']
    categorical_cols = list(encoder.feature_names_in_)

    # Create a copy of the input data to avoid modifying the original
    processed_data = new_data.copy()

    # Scale numeric data if a scaler is provided
    if scaler is not None:
        processed_data[numeric_cols] = scaler.transform(processed_data[numeric_cols])

    # Encode categorical data
    encoded_data = encoder.transform(processed_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=processed_data.index)

    # Combine numeric and encoded categorical data
    final_data = pd.concat([processed_data[numeric_cols], encoded_df], axis=1)

    return final_data
# Example usage:
# df_raw = pd.read_csv('train.csv')
# preprocessed_data = preprocess_data(df_raw, scaler_numeric=True, scaler_type='standard')

# # Now, when you have new data to preprocess:
# new_data = pd.read_csv('new_data.csv')
# preprocessed_new_data = preprocess_new_data(new_data, preprocessed_data)
