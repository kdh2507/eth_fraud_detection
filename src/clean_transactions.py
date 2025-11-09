import json
import pandas as pd
from datetime import datetime

# Config
INPUT_FILE = "data/raw/raw_transactions.json"
OUTPUT_TRANSACTIONS = "data/processed/clean_transactions.csv"
OUTPUT_SUMMARY = "data/processed/data_summary.json"


def load_raw_data(filename=INPUT_FILE):
    print(f"\nLoading raw data from: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)

    print(f"  ✓ Loaded transactions from {len(data['addresses'])} addresses")
    return data


def convert_wei_to_eth(wei_value):
    try:
        return float(wei_value) / 1e18
    except:
        return 0.0


def clean_transactions(raw_data):
    """
    Steps:
    1. Flatten nested structure
    2. Remove duplicates
    3. Filter invalid transactions
    4. Standardize formats
    5. Convert units

    --> DataFrame with cleaned transactions
    """
    print("\nCleaning transactions...")

    # Flatten all transactions into a list
    all_txs = []
    for address, txs in raw_data['transactions'].items():
        if txs:
            all_txs.extend(txs)

    print(f"  Total raw transactions: {len(all_txs)}")

    df = pd.DataFrame(all_txs)

    # --- STEP 1: Remove Duplicates ---
    initial_count = len(df)
    df = df.drop_duplicates(subset=['hash'], keep='first')
    print(f"  ✓ Removed {initial_count - len(df)} duplicate transactions")

    # --- STEP 2: Filter Invalid Transactions ---
    initial_count = len(df)

    # Remove failed transactions
    df = df[df['txreceipt_status'] == '1']
    print(f"  ✓ Removed {initial_count - len(df)} failed transactions")

    initial_count = len(df)

    # df = df[df['value'] != '1']

    # Remove self-transfers
    df = df[df['from'].str.lower() != df['to'].str.lower()]
    print(f"  ✓ Removed {initial_count - len(df)} self-transfers")

    # --- STEP 3: Standardize Address Format ---
    df['from'] = df['from'].str.lower()
    df['to'] = df['to'].str.lower()
    print(f"  ✓ Standardized {len(df)} addresses to lowercase")

    # --- STEP 4: Convert Units and Data Types ---
    # Convert Wei to ETH
    df['value_eth'] = df['value'].apply(convert_wei_to_eth)
    df['gas_price_gwei'] = df['gasPrice'].apply(lambda x: float(x) / 1e9)
    df['gas_fee_eth'] = df.apply(
        lambda row: (float(row['gasUsed']) * float(row['gasPrice'])) / 1e18,
        axis=1
    )

    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timeStamp'], unit='s')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    # Convert block numbers to integers
    df['blockNumber'] = df['blockNumber'].astype(int)

    print(f"  ✓ Converted values to ETH and timestamps to datetime")

    # --- STEP 5: Add Useful Fields ---
    df['is_contract_creation'] = df['contractAddress'].notna() & (df['contractAddress'] != '')
    df['is_error'] = df['isError'] == '1'

    # --- STEP 6: Select and Order Columns ---
    columns_to_keep = [
        'hash',
        'blockNumber',
        'timestamp',
        'date',
        'hour',
        'from',
        'to',
        'value_eth',
        'gas_fee_eth',
        'gas_price_gwei',
        'gasUsed',
        'nonce',
        'transactionIndex',
        'is_contract_creation',
        'is_error',
        'confirmations'
    ]

    df_clean = df[columns_to_keep].copy()

    # Sort by timestamp
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"CLEANING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Final clean transactions: {len(df_clean)}")
    print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"Unique senders (from): {df_clean['from'].nunique()}")
    print(f"Unique receivers (to): {df_clean['to'].nunique()}")
    print(f"Total ETH transferred: {df_clean['value_eth'].sum():.2f} ETH")
    print(f"Average transaction value: {df_clean['value_eth'].mean():.4f} ETH")
    print(f"{'=' * 60}\n")

    return df_clean


def generate_data_summary(df_clean):
    summary = {
        'total_transactions': len(df_clean),
        'date_range': {
            'start': str(df_clean['date'].min()),
            'end': str(df_clean['date'].max()),
            'days': (df_clean['date'].max() - df_clean['date'].min()).days
        },
        'unique_addresses': {
            'senders': df_clean['from'].nunique(),
            'receivers': df_clean['to'].nunique(),
            'total': len(set(df_clean['from'].unique()) | set(df_clean['to'].unique()))
        },
        'transaction_values': {
            'total_eth': float(df_clean['value_eth'].sum()),
            'mean_eth': float(df_clean['value_eth'].mean()),
            'median_eth': float(df_clean['value_eth'].median()),
            'std_eth': float(df_clean['value_eth'].std()),
            'min_eth': float(df_clean['value_eth'].min()),
            'max_eth': float(df_clean['value_eth'].max())
        },
        'special_transactions': {
            'contract_creations': int(df_clean['is_contract_creation'].sum()),
            'errors': int(df_clean['is_error'].sum())
        },
        'top_senders': df_clean['from'].value_counts().head(5).to_dict(),
        'top_receivers': df_clean['to'].value_counts().head(5).to_dict()
    }

    return summary


def validate_data_quality(df_clean):
    """
    Validate cleaned data quality

    Checks:
    - No null values in critical columns
    - No negative values
    - No future timestamps
    - Address format consistency
    """
    print("\nValidating data quality...")

    issues = []

    # Check for nulls in critical columns
    critical_cols = ['hash', 'from', 'to', 'value_eth', 'timestamp']
    for col in critical_cols:
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            issues.append(f"  ✗ Found {null_count} null values in '{col}'")

    # Check for negative values
    if (df_clean['value_eth'] < 0).any():
        issues.append(f"  ✗ Found negative values in 'value_eth'")

    # Check for future timestamps
    if (df_clean['timestamp'] > datetime.now()).any():
        issues.append(f"  ✗ Found future timestamps")

    # Check address format (should all be lowercase and start with 0x)
    invalid_from = ~df_clean['from'].str.match(r'^0x[a-f0-9]{40}$')
    invalid_to = ~df_clean['to'].str.match(r'^0x[a-f0-9]{40}$')

    if invalid_from.any():
        issues.append(f"  ✗ Found {invalid_from.sum()} invalid 'from' addresses")
    if invalid_to.any():
        issues.append(f"  ✗ Found {invalid_to.sum()} invalid 'to' addresses")

    if issues:
        print("\n⚠️  Data Quality Issues Found:")
        for issue in issues:
            print(issue)
    else:
        print("  ✓ All data quality checks passed!")

    return len(issues) == 0


def save_clean_data(df_clean, summary,
                    output_csv=OUTPUT_TRANSACTIONS,
                    output_summary=OUTPUT_SUMMARY):
    """
    Save cleaned data and summary

    Args:
        df_clean: Cleaned transactions DataFrame
        summary: Summary statistics dictionary
        output_csv: Output CSV filename
        output_summary: Output JSON filename for summary
    """
    # Save transactions
    df_clean.to_csv(output_csv, index=False)
    print(f"\n✓ Clean transactions saved to: {output_csv}")

    # Save summary
    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Data summary saved to: {output_summary}")

    print(f"\n{'=' * 60}")
    print("DATA CLEANING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Clean transactions: {len(df_clean)}")
    print(f"Ready for Phase 3: Graph Construction")
    print(f"{'=' * 60}\n")


def main():
    """
    Main function to orchestrate the cleaning process
    """
    print("\n" + "=" * 60)
    print("TRANSACTION DATA CLEANING - PHASE 2")
    print("=" * 60)

    # Load raw data
    raw_data = load_raw_data()

    # Clean transactions
    df_clean = clean_transactions(raw_data)

    # Generate summary
    summary = generate_data_summary(df_clean)

    # Validate quality
    is_valid = validate_data_quality(df_clean)

    # Save results
    save_clean_data(df_clean, summary)

    if is_valid:
        print("✓ All validation checks passed!")
        print("  Data is ready for graph construction.")
    else:
        print("⚠️  Please review data quality issues before proceeding.")


if __name__ == "__main__":
    main()