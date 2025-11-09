"""
Build transaction graph from clean data
Creates node list (addresses) and edge list (transactions)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import networkx as nx

# Configuration
INPUT_FILE = "data/processed/clean_transactions.csv"
OUTPUT_NODES = "data/processed/graph_nodes.csv"
OUTPUT_EDGES = "data/processed/graph_edges.csv"
OUTPUT_GRAPH_INFO = "data/processed/graph_info.json"


def load_clean_transactions(filename=INPUT_FILE):
    """
    Load cleaned transaction data

    Returns:
        DataFrame with clean transactions
    """
    print(f"\nLoading clean transactions from: {filename}")
    df = pd.read_csv(filename, parse_dates=['timestamp', 'date'])
    print(f"  ✓ Loaded {len(df)} transactions")
    return df


def create_node_list(df):
    """
    Extract unique addresses and create node list
    Each unique wallet address becomes a node

    Returns:
        DataFrame with node information
    """
    print("\nCreating node list...")

    # Get all unique addresses (both senders and receivers)
    all_addresses = set(df['from'].unique()) | set(df['to'].unique())
    print(f"  Total unique addresses: {len(all_addresses)}")

    # Create node dataframe
    address_list = [addr for addr in all_addresses if isinstance(addr, str)]
    nodes = pd.DataFrame({
        'address': sorted(address_list)
    })

    # Assign unique node IDs
    nodes['node_id'] = range(len(nodes))

    # Calculate basic statistics for each node
    print("  Calculating node statistics...")

    # Incoming transactions
    incoming = df.groupby('to').agg({
        'hash': 'count',
        'value_eth': 'sum',
        'timestamp': ['min', 'max']
    }).reset_index()
    incoming.columns = ['address', 'in_degree', 'total_received', 'first_received', 'last_received']

    # Outgoing transactions
    outgoing = df.groupby('from').agg({
        'hash': 'count',
        'value_eth': 'sum',
        'timestamp': ['min', 'max']
    }).reset_index()
    outgoing.columns = ['address', 'out_degree', 'total_sent', 'first_sent', 'last_sent']

    # Merge statistics
    nodes = nodes.merge(incoming, on='address', how='left')
    nodes = nodes.merge(outgoing, on='address', how='left')

    # Fill NaN values with 0 for addresses that only sent or only received
    nodes['in_degree'] = nodes['in_degree'].fillna(0).astype(int)
    nodes['out_degree'] = nodes['out_degree'].fillna(0).astype(int)
    nodes['total_received'] = nodes['total_received'].fillna(0)
    nodes['total_sent'] = nodes['total_sent'].fillna(0)

    # Calculate total degree
    nodes['total_degree'] = nodes['in_degree'] + nodes['out_degree']

    # Calculate first and last activity
    nodes['first_activity'] = nodes[['first_received', 'first_sent']].min(axis=1)
    nodes['last_activity'] = nodes[['last_received', 'last_sent']].max(axis=1)

    # Calculate activity duration in days
    nodes['activity_days'] = (nodes['last_activity'] - nodes['first_activity']).dt.days
    nodes['activity_days'] = nodes['activity_days'].fillna(0)

    # Net balance (received - sent)
    nodes['net_balance'] = nodes['total_received'] - nodes['total_sent']

    # Node type classification
    nodes['node_type'] = 'normal'
    nodes.loc[nodes['in_degree'] == 0, 'node_type'] = 'source_only'  # Only sends
    nodes.loc[nodes['out_degree'] == 0, 'node_type'] = 'sink_only'  # Only receives
    nodes.loc[(nodes['in_degree'] > 50) | (nodes['out_degree'] > 50), 'node_type'] = 'hub'  # High activity

    print(f"  ✓ Created {len(nodes)} nodes")
    print(f"    - Source only: {(nodes['node_type'] == 'source_only').sum()}")
    print(f"    - Sink only: {(nodes['node_type'] == 'sink_only').sum()}")
    print(f"    - Hubs: {(nodes['node_type'] == 'hub').sum()}")
    print(f"    - Normal: {(nodes['node_type'] == 'normal').sum()}")

    return nodes


def create_edge_list(df, nodes):
    """
    Create edge list from transactions
    Each transaction becomes a directed edge

    Args:
        df: Transaction DataFrame
        nodes: Node DataFrame with address-to-ID mapping

    Returns:
        DataFrame with edge information
    """
    print("\nCreating edge list...")

    # Create address to node_id mapping
    address_to_id = dict(zip(nodes['address'], nodes['node_id']))

    # Map addresses to node IDs
    df['source_id'] = df['from'].map(address_to_id)
    df['target_id'] = df['to'].map(address_to_id)

    # Create edge dataframe
    edges = df[[
        'hash',
        'source_id',
        'target_id',
        'timestamp',
        'date',
        'hour',
        'value_eth',
        'gas_fee_eth',
        'gas_price_gwei'
    ]].copy()

    # Rename for clarity
    edges = edges.rename(columns={
        'hash': 'transaction_hash'
    })

    # Add edge ID
    edges['edge_id'] = range(len(edges))

    # Add temporal features
    edges['day_of_week'] = pd.to_datetime(edges['timestamp']).dt.dayofweek
    edges['month'] = pd.to_datetime(edges['timestamp']).dt.month

    # Reorder columns
    edges = edges[[
        'edge_id',
        'source_id',
        'target_id',
        'transaction_hash',
        'timestamp',
        'date',
        'hour',
        'day_of_week',
        'month',
        'value_eth',
        'gas_fee_eth',
        'gas_price_gwei'
    ]]

    print(f"  ✓ Created {len(edges)} edges")

    return edges


def assign_time_steps(edges, window='1D'):
    """
    Assign time steps to edges for temporal graph analysis
    Groups transactions into discrete time windows

    Args:
        edges: Edge DataFrame
        window: Time window ('1H'=hour, '1D'=day, '1W'=week)

    Returns:
        Edge DataFrame with time_step column
    """
    print(f"\nAssigning time steps (window: {window})...")

    # Sort by timestamp
    edges = edges.sort_values('timestamp').copy()

    # Get min timestamp
    min_time = edges['timestamp'].min()

    # Calculate time step for each edge
    if window == '1H':
        edges['time_step'] = ((edges['timestamp'] - min_time).dt.total_seconds() / 3600).astype(int)
    elif window == '1D':
        edges['time_step'] = ((edges['timestamp'] - min_time).dt.total_seconds() / 86400).astype(int)
    elif window == '1W':
        edges['time_step'] = ((edges['timestamp'] - min_time).dt.total_seconds() / 604800).astype(int)
    else:
        edges['time_step'] = 0

    num_time_steps = edges['time_step'].max() + 1
    print(f"  ✓ Created {num_time_steps} time steps")

    return edges


def calculate_graph_statistics(nodes, edges):
    """
    Calculate graph-level statistics using NetworkX

    Returns:
        Dictionary with graph statistics
    """
    print("\nCalculating graph statistics...")

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(nodes['node_id'].tolist())

    # Add edges
    edge_list = list(zip(edges['source_id'], edges['target_id']))
    G.add_edges_from(edge_list)

    # Calculate statistics
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G),
        'num_weakly_connected_components': nx.number_weakly_connected_components(G),
        'num_strongly_connected_components': nx.number_strongly_connected_components(G),
    }

    # Largest component
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    stats['largest_component_size'] = len(largest_wcc)
    stats['largest_component_percentage'] = len(largest_wcc) / G.number_of_nodes() * 100

    # Degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    stats['avg_in_degree'] = np.mean(in_degrees)
    stats['avg_out_degree'] = np.mean(out_degrees)
    stats['max_in_degree'] = np.max(in_degrees)
    stats['max_out_degree'] = np.max(out_degrees)

    print(f"  ✓ Graph statistics calculated")

    return stats, G


def analyze_graph_structure(G, nodes, top_k=10):
    """
    Analyze graph structure and find important nodes

    Returns:
        Dictionary with analysis results
    """
    print("\nAnalyzing graph structure...")

    analysis = {}

    # Find nodes with highest degrees
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())

    top_in = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_out = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Get addresses for top nodes
    node_id_to_address = dict(zip(nodes['node_id'], nodes['address']))

    analysis['top_receivers'] = [
        {'address': node_id_to_address[node_id], 'in_degree': degree}
        for node_id, degree in top_in
    ]

    analysis['top_senders'] = [
        {'address': node_id_to_address[node_id], 'out_degree': degree}
        for node_id, degree in top_out
    ]

    print(f"  ✓ Identified top {top_k} senders and receivers")

    return analysis


def save_graph_data(nodes, edges, stats, analysis,
                    output_nodes=OUTPUT_NODES,
                    output_edges=OUTPUT_EDGES,
                    output_info=OUTPUT_GRAPH_INFO):
    """
    Save graph data to files
    """
    print("\nSaving graph data...")

    # Save nodes
    nodes.to_csv(output_nodes, index=False)
    print(f"  ✓ Nodes saved to: {output_nodes}")

    # Save edges
    edges.to_csv(output_edges, index=False)
    print(f"  ✓ Edges saved to: {output_edges}")

    # Combine stats and analysis
    graph_info = {
        'creation_date': datetime.now().isoformat(),
        'statistics': stats,
        'analysis': analysis,
        'files': {
            'nodes': output_nodes,
            'edges': output_edges
        }
    }

    # Save graph info
    with open(output_info, 'w') as f:
        json.dump(graph_info, f, indent=2, default=str)
    print(f"  ✓ Graph info saved to: {output_info}")


def print_summary(nodes, edges, stats):
    """
    Print comprehensive summary
    """
    print(f"\n{'=' * 60}")
    print("GRAPH CONSTRUCTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Nodes (Addresses): {len(nodes)}")
    print(f"Edges (Transactions): {len(edges)}")
    print(f"Time Steps: {edges['time_step'].max() + 1}")
    print(f"\nGraph Density: {stats['density']:.6f}")
    print(f"Average In-Degree: {stats['avg_in_degree']:.2f}")
    print(f"Average Out-Degree: {stats['avg_out_degree']:.2f}")
    print(
        f"\nLargest Component: {stats['largest_component_size']} nodes ({stats['largest_component_percentage']:.1f}%)")
    print(f"Weakly Connected Components: {stats['num_weakly_connected_components']}")
    print(f"\nValue Statistics:")
    print(f"  Total ETH transferred: {edges['value_eth'].sum():.2f}")
    print(f"  Average transaction: {edges['value_eth'].mean():.4f} ETH")
    print(f"  Median transaction: {edges['value_eth'].median():.4f} ETH")
    print(f"{'=' * 60}\n")


def main():
    """
    Main function to orchestrate graph construction
    """
    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION - PHASE 3")
    print("=" * 60)

    # Load clean transactions
    df = load_clean_transactions()

    # Create nodes
    nodes = create_node_list(df)

    # Create edges
    edges = create_edge_list(df, nodes)

    # Assign time steps
    edges = assign_time_steps(edges, window='1D')  # Daily time steps

    # Calculate graph statistics
    stats, G = calculate_graph_statistics(nodes, edges)

    # Analyze graph structure
    analysis = analyze_graph_structure(G, nodes)

    # Save everything
    save_graph_data(nodes, edges, stats, analysis)

    # Print summary
    print_summary(nodes, edges, stats)

    print("✓ Graph construction completed!")
    print("  Ready for Phase 4: Feature Engineering")


if __name__ == "__main__":
    main()