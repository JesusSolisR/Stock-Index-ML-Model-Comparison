# --- src/visualization.py ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Set a consistent plotting style
plt.style.use('ggplot')

def plot_missing_heatmap(df_pivot):
    """
    Plots a heatmap showing missing data (NaN) across multiple indices.
    
    Args:
    df_pivot (pd.DataFrame): A "wide" format DataFrame with dates as the index
                             and stock indices as columns.
    """
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(df_pivot.isna(), cbar=False, cmap='viridis')
    plt.title('Heatmap of Missing Data per Index')
    plt.xlabel('Stock Index')
    plt.ylabel('Date')
    
    date_form = mdates.DateFormatter("%Y-%m-%d")  
    ax.yaxis.set_major_formatter(date_form)  
    plt.show()

def plot_normalized_comparison(df_pivot, start_date="2000-01-01"):
    """
    Plots a "horse race" chart comparing the normalized growth of all indices
    from a specific base date.

    Args:
    df_pivot (pd.DataFrame): A "wide" format DataFrame with dates as the index
                             and stock indices as columns.
    start_date (str):       The start date to rebase all indices to 100.
    """
    # Filter to the start date
    df_filtered = df_pivot.loc[start_date:].copy()
    
    # Fill missing values for weekends/holidays (use previous day's price)
    df_filled = df_filtered.ffill()
    
    # Clean up again (some indices might not exist until after start_date)
    df_filled.dropna(axis=1, how='all', inplace=True)
    
    # Normalize (Rebasing to 100)
    # df.iloc[0] is the first trading day's price in the filtered frame
    df_normalized = (df_filled / df_filled.iloc[0]) * 100
    
    # Plot
    plt.figure(figsize=(15, 8))
    # Plot on a matplotlib axis to get access to more functions
    ax = df_normalized.plot(figsize=(15, 8)) 
    ax.set_title(f'Normalized Stock Index Growth (Rebased to 100 at {start_date})')
    ax.set_ylabel('Normalized Price (Start = 100)')
    # Place the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout()
    plt.show()

def plot_missing_counts_by_index(df_raw_renamed):
    """
    Plots a heatmap showing the total count AND percentage of missing values (NaN)
    for each core feature (column) per stock index.

    Args:
    df_raw_renamed (pd.DataFrame): 
        The DataFrame loaded from load_raw_data(),
        which has been renamed but not yet cleaned.
    """
    
    # We only care about the core feature columns
    core_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    
    # Ensure we only select columns that exist in the DataFrame
    cols_to_check = [col for col in core_columns if col in df_raw_renamed.columns]
    
    if not cols_to_check:
         print("No core feature columns found to plot.")
         return

    # --- THIS IS THE CORRECTED LOGIC ---
    # Select the columns we want to check
    df_to_check = df_raw_renamed[cols_to_check]
    # Create a boolean (True/False) DataFrame
    df_is_na = df_to_check.isna()
    # Group this boolean DataFrame using the 'stock_index' column from the *original* df
    # and then sum the True (missing) values.
    missing_counts = df_is_na.groupby(df_raw_renamed['stock_index']).sum()
    # --- END OF CORRECTION ---
    
    # Calculate the total number of rows (days) for each index
    total_counts = df_raw_renamed.groupby('stock_index').size()
    
    # Calculate the percentage of missing values
    percentage_missing = missing_counts.div(total_counts, axis=0) * 100
    
    # (Optional) Filter out rows or columns with no missing data
    counts_filtered = missing_counts.loc[(missing_counts.sum(axis=1) > 0)]
    counts_filtered = counts_filtered.loc[:, (counts_filtered.sum() > 0)]
    
    # Apply the same filter to the percentage DataFrame
    perc_filtered = percentage_missing.loc[counts_filtered.index, counts_filtered.columns]

    if counts_filtered.empty:
        print("No missing values found across any index or feature.")
        return

    # Create the string labels (e.g., "5000\n(50.0%)")
    counts_str = counts_filtered.astype(int).astype(str)
    perc_str = perc_filtered.applymap(lambda x: f"({x:.1f}%)") # e.g., (50.0%)
    
    # Combine the two string DataFrames
    annot_labels = counts_str + "\n" + perc_str

    # Plot the heatmap
    plt.figure(figsize=(16, 12)) 
    sns.heatmap(
        counts_filtered,  # The colors are based on the *absolute counts*
        annot=annot_labels, # The labels are our new formatted strings
        fmt='',           # MUST be an empty string, since our labels are pre-formatted
        cmap='viridis_r'
    )
    plt.title('Total Missing Value Count and Percentage per Index and Feature')
    plt.xlabel('Feature Column')
    plt.ylabel('Stock Index')
    plt.show()

def plot_zero_volume_counts_by_index(df_raw_renamed):
    """
    Plots a horizontal bar chart showing the *percentage* and *count* of non-trading days (volume = 0) for each stock index.
    This provides a "fair" comparison across indices with different lifespans.

    Args:
    df_raw_renamed (pd.DataFrame): 
        The DataFrame loaded from load_raw_data(),
        which has been renamed but not yet cleaned.
    """
    
    # Calculate the total number of rows (lifespan) for each index
    total_counts = df_raw_renamed.groupby('stock_index').size()
    
    # Filter for rows where volume is exactly 0
    zero_volume_days = df_raw_renamed[df_raw_renamed['volume'] == 0]
    
    # Calculate the total count of zero-volume days each index has
    zero_counts = zero_volume_days.groupby('stock_index').size()

    # Calculate the percentage
    # We use .reindex(total_counts.index, fill_value=0) to ensure indices 
    # with 0 non-trading days are included in the calculation.
    percentage = (zero_counts.reindex(total_counts.index, fill_value=0) / total_counts) * 100
    
    # Sort by the percentage for a cleaner chart
    percentage = percentage.sort_values(ascending=True)

    if percentage.empty:
        print("No zero-volume days found for any index.")
        return

    # --- Plot the horizontal bar chart based on PERCENTAGE ---
    plt.figure(figsize=(10, 8)) 
    ax = percentage.plot(
        kind='barh', # 'h' for horizontal
        color='steelblue'
    )
    
    plt.title('Percentage of Non-Trading Days (Volume = 0) per Index')
    plt.xlabel('Percentage of Total Days')
    plt.ylabel('Stock Index')
    
    # --- Create custom labels (e.g., "3.6% (500 days)") ---
    labels = []
    for index_name in percentage.index:
        # Get the percentage value
        perc_val = percentage[index_name]
        # Get the raw count (using 0 if it doesn't exist in zero_counts)
        count_val = zero_counts.get(index_name, 0)
        labels.append(f" {perc_val:.2f}%  ({count_val} days)") # e.g., " 3.57% (500 days)"

    # Add the value labels on the end of the bars
    ax.bar_label(ax.containers[0], labels=labels, padding=5)
    
    # Adjust x-axis limits to make space for labels
    plt.xlim(0, percentage.max() * 1.15) 
    
    plt.tight_layout()
    plt.show()

def plot_zero_values(df):
    """
    Plots a bar chart showing the count of "0" values in each column
    for a single index.
    
    Args:
    df (pd.DataFrame): A "long" format DataFrame for a single index's EDA.
    """
    # Calculate zero values
    zero_values = (df == 0).sum()
    
    # Filter out columns that have no zero values to make the chart cleaner
    zero_values = zero_values[zero_values > 0]
    
    if zero_values.empty:
        print("No zero values found in any columns.")
        return
        
    # Plot
    plt.figure(figsize=(10, 6))
    zero_values.plot(kind='bar')
    plt.title('Count of Zero Values per Column')
    plt.ylabel('Number of Zero Entries')
    plt.xticks(rotation=45)
    plt.show()

def plot_distributions(df):
    """
    Plots histograms to show the data distribution of core features
    for a single index.

    Args:
    df (pd.DataFrame): A "long" format DataFrame for a single index's EDA.
    """
    core_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    
    # Filter the df to only include columns that actually exist
    cols_to_plot = [col for col in core_columns if col in df.columns]
    
    if not cols_to_plot:
        print("No core columns found to plot.")
        return

    df[cols_to_plot].hist(bins=50, figsize=(15, 10))
    plt.suptitle('Histograms of Data Distribution', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    """
    Plots boxplots to show outliers for a single index.
    Volume is plotted separately due to its different scale from prices.

    Args:
    df (pd.DataFrame): A "long" format DataFrame for a single index's EDA.
    """
    # Plot price features
    price_features = ['open', 'high', 'low', 'close', 'adj_close']
    price_features = [col for col in price_features if col in df.columns]

    if price_features:
        df[price_features].plot(kind='box', figsize=(15, 7), subplots=True, layout=(1, len(price_features)))
        plt.suptitle('Box Plots for Price Features', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # Plot Volume separately
    if 'volume' in df.columns:
        plt.figure(figsize=(6, 6))
        df[['volume']].plot(kind='box')
        plt.title('Box Plot for Volume')
        plt.show()