import requests
import json
import time
import os
from dotenv import load_dotenv
from datetime import datetime

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('ETHERSCAN_API_KEY')

# Configuration
BASE_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1  # Ethereum mainnet
OUTPUT_FILE = "data/raw/raw_transactions.json"
RATE_LIMIT_DELAY = 0.2  # 5 calls per second for free tier

ADDRESSES_TO_CRAWL = [
    "0x00000c07575Bb4e64457687A0382b4D3Ea470000",  # Scam
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # Legit
    "0xd7A38B39CeBBD766baEF23fEBC6d367DC552Bb67",  # Unknown
]


def get_transactions(address, start_block=0, end_block='latest', page=1, offset=1000):
    """
    Fetch transactions for a given address using Etherscan API V2

    Args:
        address: Ethereum wallet address
        start_block: Starting block number
        end_block: Ending block number or 'latest'
        page: Page number for pagination
        offset: Number of records per page (max 10000 for paid, 1000 for free)

    Returns:
        List of transactions or None if error
    """
    params = {
        'chainid': CHAIN_ID,
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': start_block,
        'endblock': end_block,
        'page': page,
        'offset': offset,
        'sort': 'asc',  # Chronological order
        'apikey': API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data['status'] == '1':
            return data['result']
        else:
            print(f"API Error for {address}: {data.get('message', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"Request failed for {address}: {str(e)}")
        return None


def crawl_all_transactions(address, max_transactions=10000):
    """
    Crawl all transactions for an address with pagination
    Handles the case where address has more than 1000/10000 transactions

    Args:
        address: Ethereum wallet address
        max_transactions: Maximum number of transactions to fetch

    Returns:
        List of all transactions for the address
    """
    all_transactions = []
    start_block = 0
    offset = 1000  # Free tier limit

    print(f"\n{'=' * 60}")
    print(f"Crawling transactions for address: {address}")
    print(f"{'=' * 60}")

    while len(all_transactions) < max_transactions:
        print(f"Fetching from block {start_block}...")

        transactions = get_transactions(
            address=address,
            start_block=start_block,
            end_block='latest',
            offset=offset
        )

        if transactions is None:
            break

        if len(transactions) == 0:
            print("No more transactions found.")
            break

        all_transactions.extend(transactions)
        print(f"  Retrieved {len(transactions)} transactions (Total: {len(all_transactions)})")

        # If we got less than offset, we've reached the end
        if len(transactions) < offset:
            print("Reached end of transaction history.")
            break

        # Update start_block for next iteration
        # Set to last block number - 1 to avoid missing transactions
        last_block = int(transactions[-1]['blockNumber'])
        start_block = last_block - 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Safety limit
        if len(all_transactions) >= max_transactions:
            print(f"Reached maximum transaction limit ({max_transactions})")
            break

    return all_transactions


def save_raw_data(all_data, filename=OUTPUT_FILE):
    """
    Save raw transaction data to JSON file

    Args:
        all_data: Dictionary containing all crawled data
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Raw data saved to: {filename}")
    print(f"Total addresses crawled: {len(all_data['addresses'])}")
    print(f"Total transactions: {sum(len(v) for v in all_data['transactions'].values())}")
    print(f"{'=' * 60}\n")


def main():
    """
    Main function to orchestrate the crawling process
    """
    print("\n" + "=" * 60)
    print("BLOCKCHAIN TRANSACTION CRAWLER - PHASE 1")
    print("Etherscan API V2 - Ethereum Mainnet")
    print("=" * 60)

    if not API_KEY:
        print("ERROR: ETHERSCAN_API_KEY not found in .env file!")
        print("Please create a .env file with: ETHERSCAN_API_KEY=your_key_here")
        return

    # Data structure to store all crawled data
    crawled_data = {
        'metadata': {
            'crawl_date': datetime.now().isoformat(),
            'chain_id': CHAIN_ID,
            'chain_name': 'Ethereum Mainnet',
            'api_version': 'v2'
        },
        'addresses': ADDRESSES_TO_CRAWL,
        'transactions': {}
    }

    # Crawl transactions for each address
    for address in ADDRESSES_TO_CRAWL:
        transactions = crawl_all_transactions(address, max_transactions=5000)
        crawled_data['transactions'][address] = transactions

        # Small delay between addresses
        time.sleep(1)

    # Save raw data
    save_raw_data(crawled_data)

    print("\nSuccess!")


if __name__ == "__main__":
    main()