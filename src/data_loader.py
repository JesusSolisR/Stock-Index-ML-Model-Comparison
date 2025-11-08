import pandas as pd

def load_raw_data(stock_index=None, start_date=None, end_date=None, file_path='../data/indexData.csv'):
    """Load raw data from a given file path."""
    print(f"Loading raw data from {file_path}...")
    df_raw = pd.read_csv(file_path)
    # filter for a specific stock index if provided
    if stock_index:
        df_raw = df_raw[df_raw['Index'] == stock_index]
    # rename to follow Python's naming convention
    df = df_raw.rename(
        columns = {
            'Index': 'stock_index',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
    # make date a pandas datetime object
    df['date'] = pd.to_datetime(df['date'])
    # filter for date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    return df

def load_data(stock_index=None, start_date=None, end_date=None, file_path='../data/indexData.csv'):
    """Load data from a given file path."""
    """start_date and end_date should be in 'YYYY-MM-DD' format strings."""
    """file_path is defaulted to '../data/indexData.csv'."""
    print(f"Loading data from {file_path}...")
    df = load_raw_data(stock_index, start_date, end_date, file_path)
    # set date as index and sort by date as it's time series data
    df = df.set_index('date')
    df.sort_index(inplace=True)
    # drop rows with missing core data
    original_rows = len(df)
    core_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df.dropna(subset=core_columns, inplace=True)
    print(f"Dropped {original_rows - len(df)} rows due to missing values.")
    # 6. Drop any row where volume was 0 (non-trading day)
    original_rows = len(df)
    df = df[df['volume'] > 0]
    print(f"Dropped {original_rows - len(df)} rows due to zero volume.")
    return df