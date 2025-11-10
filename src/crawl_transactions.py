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
    # Your originals
    "0x00000c07575Bb4e64457687A0382b4D3Ea470000",  # Scam
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # Legit (WETH)
    "0xd7A38B39CeBBD766baEF23fEBC6d367DC552Bb67",  # Unknown

    # Illicit (Etherscan-tagged phishing/scam)
    "0xd9a1c3788d81257612e2581a6ea0ada244853a91",  # Scam
    "0x474057adf42f9f955e86aa1142740f9d7763e41e",  # Scam
    "0xd13b093ea3a0b9b7a2f3b2a4a0e6cc2d2b0abf9e",  # Scam
    "0x000011389c8a3a3e4ce1b1d8b4c1ef2f0efd3b0000",  # Scam

    # Licit (exchanges, tokens, routers)
    "0xF977814e90dA44bFA03b6295A0616a897441aceC",  # Binance Hot Wallet 20
    "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf",  # Kraken 13
    "0xBE0eB53F46cd790cd13851d5EFf43D12404d33E8",  # Binance 7
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0",  # Kraken 4
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # Uniswap V3 Router 2
    "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # Tether USDT
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # Circle USDC
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