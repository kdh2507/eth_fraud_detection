"""
Phase 4: Advanced Feature Engineering
Calculate graph-based features for GNN training
Includes: centrality, clustering, PageRank, temporal features, etc.
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_NODES = "data/processed/graph_nodes.csv"
INPUT_EDGES = "data/processed/graph_edges.csv"
OUTPUT_FEATURES = "data/processed/node_features.csv"
OUTPUT_EDGE_FEATURES = "data/processed/edge_features.csv"


def load_graph_data(nodes_file=INPUT_NODES, edges_file=INPUT_EDGES):
    """
    Load graph nodes and edges
    """
    print(f"\nLoading graph data...")
    nodes = pd.read_csv(nodes_file, parse_dates=['first_activity', 'last_activity'])
    edges = pd.read_csv(edges_file, parse_dates=['timestamp', 'date'])

    print(f"  ✓ Loaded {len(nodes)} nodes and {len(edges)} edges")
    return nodes, edges


def build_networkx_graph(nodes, edges):
    """
    Build NetworkX graph for centrality calculations
    """
    print("\nBuilding NetworkX graph...")
    G = nx.DiGraph()
    G.add_nodes_from(nodes['node_id'].tolist())

    # Add edges with weights (transaction values)
    for _, edge in edges.iterrows():
        G.add_edge(
            edge['source_id'],
            edge['target_id'],
            weight=edge['value_eth']
        )

    print(f"  ✓ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def calculate_centrality_features(G, nodes):
    """
    Calculate various centrality measures
    These identify important/influential nodes in the network
    """
    print("\nCalculating centrality features...")
    features = pd.DataFrame({'node_id': nodes['node_id'].unique()})

    # Degree centrality (normalized by max possible degree)
    print("  - Degree centrality...")
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)
    features['in_degree_centrality'] = features['node_id'].apply(lambda x: in_degree_cent.get(x, 0))
    features['out_degree_centrality'] = features['node_id'].apply(lambda x: out_degree_cent.get(x, 0))

    # PageRank (importance based on incoming connections)
    print("  - PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    features['pagerank'] = features['node_id'].apply(lambda x: pagerank.get(x, 0))

    # Betweenness centrality (how often node appears on shortest paths)
    # Note: This is computationally expensive, use sampling for large graphs
    if len(G.nodes()) < 1000:
        print("  - Betweenness centrality...")
        betweenness = nx.betweenness_centrality(G, normalized=True)
        features['betweenness_centrality'] = features['node_id'].apply(lambda x: betweenness.get(x, 0))
    else:
        print("  - Betweenness centrality (sampled)...")
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes())), normalized=True)
        features['betweenness_centrality'] = features['node_id'].apply(lambda x: betweenness.get(x, 0))

    # Closeness centrality (average distance to all other nodes)
    print("  - Closeness centrality...")
    try:
        closeness = nx.closeness_centrality(G)
        features['closeness_centrality'] = features['node_id'].apply(lambda x: closeness.get(x, 0))
    except:
        # If graph is not connected, set to 0
        features['closeness_centrality'] = 0

    print(f"  ✓ Calculated centrality features")
    return features


def calculate_clustering_features(G, nodes):
    """
    Calculate clustering coefficient
    Measures how much nodes cluster together
    """
    print("\nCalculating clustering features...")

    # Clustering coefficient (for undirected version)
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    features = pd.DataFrame({
        'node_id': nodes['node_id'].unique(),
        'clustering_coefficient': 0.0
    })
    features['clustering_coefficient'] = features['node_id'].apply(lambda x: clustering.get(x, 0))

    print(f"  ✓ Calculated clustering coefficient")
    return features


def calculate_local_structure_features(G, nodes):
    """
    Calculate features about local neighborhood structure
    """
    print("\nCalculating local structure features...")

    features = pd.DataFrame({'node_id': nodes['node_id'].unique()})

    # Number of neighbors at different hops
    print("  - Ego network sizes...")
    unique_nodes = nodes['node_id'].unique()
    for node_id in unique_nodes:
        # 1-hop neighbors
        neighbors_1 = set(G.predecessors(node_id)) | set(G.successors(node_id))
        features.loc[features['node_id'] == node_id, 'neighbors_1hop'] = len(neighbors_1)

        # 2-hop neighbors
        neighbors_2 = set()
        for n in neighbors_1:
            neighbors_2.update(G.predecessors(n))
            neighbors_2.update(G.successors(n))
        neighbors_2 -= neighbors_1
        neighbors_2.discard(node_id)
        features.loc[features['node_id'] == node_id, 'neighbors_2hop'] = len(neighbors_2)

    # Triangle count (number of triangles node participates in)
    print("  - Triangle counts...")
    G_undirected = G.to_undirected()
    triangles = nx.triangles(G_undirected)
    features['triangle_count'] = features['node_id'].apply(lambda x: triangles.get(x, 0))

    print(f"  ✓ Calculated local structure features")
    return features


def calculate_temporal_features(edges, nodes):
    """
    Calculate temporal behavior features
    """
    print("\nCalculating temporal features...")

    features = pd.DataFrame({'node_id': nodes['node_id'].unique()})

    # For each node, calculate temporal patterns
    unique_nodes = nodes['node_id'].unique()
    for node_id in unique_nodes:
        # Incoming transactions
        incoming = edges[edges['target_id'] == node_id].sort_values('timestamp')
        # Outgoing transactions
        outgoing = edges[edges['source_id'] == node_id].sort_values('timestamp')

        # Transaction frequency (transactions per day)
        if len(incoming) > 0:
            in_days = (incoming['timestamp'].max() - incoming['timestamp'].min()).days + 1
            features.loc[features['node_id'] == node_id, 'in_tx_per_day'] = len(incoming) / max(in_days, 1)

            # Average time between incoming transactions (in hours)
            if len(incoming) > 1:
                time_diffs = incoming['timestamp'].diff().dt.total_seconds() / 3600
                features.loc[features['node_id'] == node_id, 'avg_time_between_in_tx'] = time_diffs.mean()

        if len(outgoing) > 0:
            out_days = (outgoing['timestamp'].max() - outgoing['timestamp'].min()).days + 1
            features.loc[features['node_id'] == node_id, 'out_tx_per_day'] = len(outgoing) / max(out_days, 1)

            # Average time between outgoing transactions (in hours)
            if len(outgoing) > 1:
                time_diffs = outgoing['timestamp'].diff().dt.total_seconds() / 3600
                features.loc[features['node_id'] == node_id, 'avg_time_between_out_tx'] = time_diffs.mean()

        # Burst detection (std of transaction times)
        if len(outgoing) > 1:
            time_diffs = outgoing['timestamp'].diff().dt.total_seconds()
            features.loc[features['node_id'] == node_id, 'tx_time_std'] = time_diffs.std()

    # Fill NaN with 0
    features = features.fillna(0)

    print(f"  ✓ Calculated temporal features")
    return features


def calculate_transaction_pattern_features(edges, nodes):
    """
    Calculate features based on transaction patterns
    """
    print("\nCalculating transaction pattern features...")

    features = pd.DataFrame({'node_id': nodes['node_id'].unique()})

    unique_nodes = nodes['node_id'].unique()
    for node_id in unique_nodes:
        # Incoming patterns
        incoming = edges[edges['target_id'] == node_id]
        # Outgoing patterns
        outgoing = edges[edges['source_id'] == node_id]

        # Unique counterparties
        unique_senders = incoming['source_id'].nunique()
        unique_receivers = outgoing['target_id'].nunique()
        features.loc[features['node_id'] == node_id, 'unique_senders'] = unique_senders
        features.loc[features['node_id'] == node_id, 'unique_receivers'] = unique_receivers

        # Repetition rate (how often transacts with same addresses)
        if len(incoming) > 0:
            sender_counts = incoming['source_id'].value_counts()
            repetition_in = (sender_counts > 1).sum() / len(sender_counts)
            features.loc[features['node_id'] == node_id, 'repetition_rate_in'] = repetition_in

        if len(outgoing) > 0:
            receiver_counts = outgoing['target_id'].value_counts()
            repetition_out = (receiver_counts > 1).sum() / len(receiver_counts)
            features.loc[features['node_id'] == node_id, 'repetition_rate_out'] = repetition_out

        # Fan-out ratio (one-to-many vs many-to-one)
        total_tx = len(incoming) + len(outgoing)
        if total_tx > 0:
            fan_out_ratio = len(outgoing) / total_tx
            features.loc[features['node_id'] == node_id, 'fan_out_ratio'] = fan_out_ratio

        # Value patterns
        if len(incoming) > 0:
            features.loc[features['node_id'] == node_id, 'avg_received_value'] = incoming['value_eth'].mean()
            features.loc[features['node_id'] == node_id, 'std_received_value'] = incoming['value_eth'].std()
            features.loc[features['node_id'] == node_id, 'max_received_value'] = incoming['value_eth'].max()

        if len(outgoing) > 0:
            features.loc[features['node_id'] == node_id, 'avg_sent_value'] = outgoing['value_eth'].mean()
            features.loc[features['node_id'] == node_id, 'std_sent_value'] = outgoing['value_eth'].std()
            features.loc[features['node_id'] == node_id, 'max_sent_value'] = outgoing['value_eth'].max()

        # Time of day patterns
        if len(outgoing) > 0:
            hour_counts = outgoing['hour'].value_counts()
            features.loc[features['node_id'] == node_id, 'most_active_hour'] = hour_counts.idxmax()
            # Activity concentration (higher = more concentrated in specific hours)
            features.loc[features['node_id'] == node_id, 'hour_concentration'] = (hour_counts.max() / len(outgoing))

    # Fill NaN with 0
    features = features.fillna(0)

    print(f"  ✓ Calculated transaction pattern features")
    return features


def calculate_edge_features(edges):
    """
    Calculate features for each edge (transaction)
    """
    print("\nCalculating edge features...")

    edge_features = edges[['edge_id', 'source_id', 'target_id']].copy()

    # Normalize transaction value (0-1 scale)
    edge_features['value_normalized'] = (edges['value_eth'] - edges['value_eth'].min()) / \
                                        (edges['value_eth'].max() - edges['value_eth'].min() + 1e-10)

    # Time features
    edge_features['hour'] = edges['hour']
    edge_features['day_of_week'] = edges['day_of_week']
    edge_features['month'] = edges['month']
    edge_features['time_step'] = edges['time_step']

    # Transaction size category
    edge_features['tx_size_category'] = pd.cut(
        edges['value_eth'],
        bins=[0, 0.01, 0.1, 1, 10, float('inf')],
        labels=['micro', 'small', 'medium', 'large', 'very_large']
    )

    # Gas price category (indicates urgency)
    edge_features['gas_price_category'] = pd.cut(
        edges['gas_price_gwei'],
        bins=[0, 20, 50, 100, float('inf')],
        labels=['low', 'normal', 'high', 'very_high']
    )

    print(f"  ✓ Calculated edge features for {len(edge_features)} edges")
    return edge_features


def combine_all_features(nodes, centrality_feat, clustering_feat, local_feat,
                         temporal_feat, pattern_feat):
    """
    Combine all feature sets into final feature matrix
    """
    print("\nCombining all features...")

    # Start with basic node features
    features = nodes[['node_id', 'address', 'in_degree', 'out_degree', 'total_degree',
                      'total_received', 'total_sent', 'net_balance', 'activity_days',
                      'node_type']].copy()

    # Merge all feature sets
    features = features.merge(centrality_feat, on='node_id', how='left')
    features = features.merge(clustering_feat, on='node_id', how='left')
    features = features.merge(local_feat, on='node_id', how='left')
    features = features.merge(temporal_feat, on='node_id', how='left')
    features = features.merge(pattern_feat, on='node_id', how='left')

    # Fill any remaining NaN with 0
    features = features.fillna(0)

    print(f"  ✓ Final feature matrix: {features.shape[0]} nodes × {features.shape[1]} features")
    return features


def save_features(node_features, edge_features):
    """
    Save feature matrices
    """
    print("\nSaving features...")

    # Save node features
    node_features.to_csv(OUTPUT_FEATURES, index=False)
    print(f"  ✓ Node features saved to: {OUTPUT_FEATURES}")

    # Save edge features
    edge_features.to_csv(OUTPUT_EDGE_FEATURES, index=False)
    print(f"  ✓ Edge features saved to: {OUTPUT_EDGE_FEATURES}")

    print(f"\n{'=' * 60}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Node features: {node_features.shape[0]} nodes × {node_features.shape[1]} features")
    print(f"Edge features: {edge_features.shape[0]} edges × {edge_features.shape[1]} features")
    print(f"\nNode feature categories:")
    print(f"  - Basic statistics: 9 features")
    print(f"  - Centrality measures: 5 features")
    print(f"  - Clustering: 1 feature")
    print(f"  - Local structure: 3 features")
    print(f"  - Temporal patterns: 5 features")
    print(f"  - Transaction patterns: 12 features")
    print(f"{'=' * 60}\n")


def main():
    """
    Main function to orchestrate feature engineering
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING - PHASE 4")
    print("=" * 60)

    # Load graph data
    nodes, edges = load_graph_data()

    # Build NetworkX graph
    G = build_networkx_graph(nodes, edges)

    # Calculate different feature sets
    centrality_feat = calculate_centrality_features(G, nodes)
    clustering_feat = calculate_clustering_features(G, nodes)
    local_feat = calculate_local_structure_features(G, nodes)
    temporal_feat = calculate_temporal_features(edges, nodes)
    pattern_feat = calculate_transaction_pattern_features(edges, nodes)

    # Calculate edge features
    edge_features = calculate_edge_features(edges)

    # Combine all node features
    node_features = combine_all_features(
        nodes, centrality_feat, clustering_feat,
        local_feat, temporal_feat, pattern_feat
    )

    # Save everything
    save_features(node_features, edge_features)

    print("✓ Feature engineering completed!")
    print("  Ready for Phase 5: Label Assignment")


if __name__ == "__main__":
    main()