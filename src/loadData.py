import pandas as pd

def load_data(file_path='../data/indexData.csv', stock_index = None):
    """Load data from a given file path."""
    print(f"Loading data from {file_path}...")
    df_raw = pd.read_csv(file_path)
    # 1. filter for a specific stock index if provided
    if stock_index:
        df_raw = df_raw[df_raw['Index'] == stock_index]
    # 2. rename to follow Python's naming convention
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
    # 3. make date a pandas datetime object
    df['date'] = pd.to_datetime(df['date'])
    # 4. set date as index and sort by date as it's time series data
    df = df.set_index('date')
    df.sort_index(inplace=True)
    # 5. drop rows with missing core data
    original_rows = len(df)
    core_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df.dropna(subset=core_columns, inplace=True)
    print(f"Dropped {original_rows - len(df)} rows due to missing values.")
    # 6. Drop any row where volume was 0 (non-trading day)
    original_rows = len(df)
    df = df[df['volume'] > 0]
    print(f"Dropped {original_rows - len(df)} rows due to zero volume.")
    return df