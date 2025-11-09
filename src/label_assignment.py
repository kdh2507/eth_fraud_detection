"""
Phase 5: Label Assignment
Assign fraud/non-fraud labels to nodes using heuristics and manual lists
For real blockchain data, we need to create labels ourselves
"""

import pandas as pd
import numpy as np
import json

# Configuration
INPUT_FEATURES = "data/processed/node_features.csv"
INPUT_EDGES = "data/processed/graph_edges.csv"
OUTPUT_LABELED = "data/processed/labeled_nodes.csv"
OUTPUT_LABEL_STATS = "data/processed/label_statistics.json"

# Known addresses (you can expand these lists)
# These are examples - replace with actual blacklist data
KNOWN_ILLICIT_ADDRESSES = [
    # Add known scam/fraud addresses here
    # Example: "0x1234567890abcdef1234567890abcdef12345678"
    "0x00000c07575Bb4e64457687A0382b4D3Ea470000",
    "0x474057adf42f9f955e86aa1142740f9d7763e41e",
    "0x000011389c8a3a3e4ce1b1d8b4c1ef2f0efd3b0000",
    "0xd9a1c3788d81257612e2581a6ea0ada244853a91",
    "0xd13b093ea3a0b9b7a2f3b2a4a0e6cc2d2b0abf9e"
]

KNOWN_LEGITIMATE_ADDRESSES = [
    # Add known legitimate addresses (exchanges, etc.)
    # Example: Binance, Coinbase addresses
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "0xd13b093ea3a0b9b7a2f3b2a4a0e6cc2d2b0abf9e",
    "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf",
    "0x6d0cf1f6a8e8d2fd5ccdbb4c1f6d7d8389e93d26",
    "0xe9f7ecae7a0d4c6a6de72d2f8d4b9a70d1fab785",

]


def load_features(filename=INPUT_FEATURES):
    """
    Load node features
    """
    print(f"\nLoading node features from: {filename}")
    features = pd.read_csv(filename)
    print(f"  ✓ Loaded {len(features)} nodes")
    return features


def load_edges(filename=INPUT_EDGES):
    """
    Load edges for pattern analysis
    """
    print(f"Loading edges from: {filename}")
    edges = pd.read_csv(filename, parse_dates=['timestamp', 'date'])
    print(f"  ✓ Loaded {len(edges)} edges")
    return edges


def assign_known_labels(features):
    """
    Assign labels based on known addresses
    0 = Unknown, 1 = Illicit, 2 = Licit
    """
    print("\nAssigning known labels...")

    features['label'] = 0  # Default: Unknown
    features['label_source'] = 'unknown'
    features['label_confidence'] = 0.0

    # Assign illicit labels
    illicit_mask = features['address'].isin(KNOWN_ILLICIT_ADDRESSES)
    features.loc[illicit_mask, 'label'] = 1
    features.loc[illicit_mask, 'label_source'] = 'blacklist'
    features.loc[illicit_mask, 'label_confidence'] = 1.0

    # Assign licit labels
    licit_mask = features['address'].isin(KNOWN_LEGITIMATE_ADDRESSES)
    features.loc[licit_mask, 'label'] = 2
    features.loc[licit_mask, 'label_source'] = 'whitelist'
    features.loc[licit_mask, 'label_confidence'] = 1.0

    print(f"  ✓ Labeled {illicit_mask.sum()} as illicit (from blacklist)")
    print(f"  ✓ Labeled {licit_mask.sum()} as licit (from whitelist)")

    return features


def apply_heuristic_suspicious_patterns(features, edges):
    """
    Apply heuristic rules to identify potentially suspicious addresses
    These are "soft" labels with lower confidence
    """
    print("\nApplying heuristic rules for suspicious patterns...")

    suspicious_count = 0

    for idx, node in features.iterrows():
        # Skip if already labeled
        if features.loc[idx, 'label'] != 0:
            continue

        node_id = node['node_id']
        suspicion_score = 0
        suspicion_reasons = []

        # Get transactions for this node
        node_edges_out = edges[edges['source_id'] == node_id]
        node_edges_in = edges[edges['target_id'] == node_id]

        # Heuristic 1: Very short-lived address with high value
        if node['activity_days'] <= 1 and node['total_sent'] > 10:
            suspicion_score += 0.3
            suspicion_reasons.append('short_lived_high_value')

        # Heuristic 2: Rapid fund dispersion (receive then immediately send to many)
        if len(node_edges_out) > 0 and len(node_edges_in) > 0:
            # Check if outgoing transactions happen quickly after incoming
            first_in = node_edges_in['timestamp'].min()
            first_out = node_edges_out['timestamp'].min()
            if pd.notna(first_in) and pd.notna(first_out):
                time_diff = (first_out - first_in).total_seconds() / 3600  # hours
                if time_diff < 24 and node['unique_receivers'] > 10:
                    suspicion_score += 0.3
                    suspicion_reasons.append('rapid_dispersion')

        # Heuristic 3: One-time large receiver (sink only with large amount)
        if node['node_type'] == 'sink_only' and node['in_degree'] == 1 and node['total_received'] > 5:
            suspicion_score += 0.2
            suspicion_reasons.append('one_time_sink')

        # Heuristic 4: High repetition rate (always transacts with same few addresses)
        if node['repetition_rate_out'] > 0.8 and node['out_degree'] > 5:
            suspicion_score += 0.2
            suspicion_reasons.append('high_repetition')

        # Heuristic 5: Unusual time patterns (mostly night activity)
        if node['most_active_hour'] >= 22 or node['most_active_hour'] <= 4:
            if node['hour_concentration'] > 0.5:
                suspicion_score += 0.1
                suspicion_reasons.append('night_activity')

        # Heuristic 6: Very high fan-out (one-to-many distribution)
        if node['fan_out_ratio'] > 0.9 and node['unique_receivers'] > 20:
            suspicion_score += 0.3
            suspicion_reasons.append('high_fanout')

        # Heuristic 7: Round number transactions (common in mixing/tumbling)
        if len(node_edges_out) > 0:
            round_count = 0
            for val in node_edges_out['value_eth']:
                # Check if value is close to round numbers (0.1, 0.5, 1, 5, 10, etc.)
                if val > 0:
                    log_val = np.log10(val)
                    if abs(log_val - round(log_val)) < 0.1:
                        round_count += 1

            if round_count / len(node_edges_out) > 0.7:
                suspicion_score += 0.2
                suspicion_reasons.append('round_numbers')

        # If suspicion score is high enough, mark as suspicious
        if suspicion_score >= 0.5:
            features.loc[idx, 'label'] = 1  # Mark as illicit
            features.loc[idx, 'label_source'] = 'heuristic: ' + ', '.join(suspicion_reasons)
            features.loc[idx, 'label_confidence'] = min(suspicion_score, 0.8)  # Max 0.8 for heuristics
            suspicious_count += 1

    print(f"  ✓ Identified {suspicious_count} suspicious addresses using heuristics")
    return features


def apply_heuristic_legitimate_patterns(features, edges):
    """
    Apply heuristic rules to identify likely legitimate addresses
    """
    print("\nApplying heuristic rules for legitimate patterns...")

    legitimate_count = 0

    for idx, node in features.iterrows():
        # Skip if already labeled
        if features.loc[idx, 'label'] != 0:
            continue

        node_id = node['node_id']
        legitimacy_score = 0
        legitimacy_reasons = []

        # Heuristic 1: Long-lived address with consistent activity
        if node['activity_days'] > 180 and node['total_degree'] > 50:
            legitimacy_score += 0.3
            legitimacy_reasons.append('long_lived')

        # Heuristic 2: Hub with balanced in/out (likely exchange or service)
        if node['node_type'] == 'hub':
            in_out_ratio = node['in_degree'] / (node['out_degree'] + 1)
            if 0.5 < in_out_ratio < 2.0:
                legitimacy_score += 0.4
                legitimacy_reasons.append('balanced_hub')

        # Heuristic 3: Many unique counterparties (diverse interactions)
        if node['unique_senders'] > 20 and node['unique_receivers'] > 20:
            legitimacy_score += 0.3
            legitimacy_reasons.append('diverse_counterparties')

        # Heuristic 4: Regular transaction patterns (consistent timing)
        if node['tx_time_std'] > 0:
            # Low std means regular intervals
            if node['tx_time_std'] < 86400:  # Less than 1 day std
                if node['out_degree'] > 10:
                    legitimacy_score += 0.2
                    legitimacy_reasons.append('regular_pattern')

        # Heuristic 5: Low repetition rate (doesn't repeat same addresses often)
        if node['repetition_rate_out'] < 0.3 and node['out_degree'] > 10:
            legitimacy_score += 0.2
            legitimacy_reasons.append('low_repetition')

        # If legitimacy score is high enough, mark as legitimate
        if legitimacy_score >= 0.6:
            features.loc[idx, 'label'] = 2  # Mark as licit
            features.loc[idx, 'label_source'] = 'heuristic: ' + ', '.join(legitimacy_reasons)
            features.loc[idx, 'label_confidence'] = min(legitimacy_score, 0.7)  # Max 0.7 for heuristics
            legitimate_count += 1

    print(f"  ✓ Identified {legitimate_count} likely legitimate addresses using heuristics")
    return features


def generate_label_statistics(features):
    """
    Generate statistics about label distribution
    """
    print("\nGenerating label statistics...")

    label_counts = features['label'].value_counts().to_dict()

    stats = {
        'total_nodes': int(len(features)),
        'labeled_nodes': int((features['label'] != 0).sum()),
        'unlabeled_nodes': int((features['label'] == 0).sum()),
        'illicit_nodes': int((features['label'] == 1).sum()),
        'licit_nodes': int((features['label'] == 2).sum()),
        'label_sources': {k: int(v) for k, v in
                          features[features['label'] != 0]['label_source'].value_counts().to_dict().items()},
        'confidence_distribution': {
            'high (>0.8)': int((features['label_confidence'] > 0.8).sum()),
            'medium (0.5-0.8)': int(
                ((features['label_confidence'] >= 0.5) & (features['label_confidence'] <= 0.8)).sum()),
            'low (<0.5)': int(((features['label_confidence'] > 0) & (features['label_confidence'] < 0.5)).sum())
        },
        'label_percentages': {
            'illicit': f"{(features['label'] == 1).sum() / len(features) * 100:.2f}%",
            'licit': f"{(features['label'] == 2).sum() / len(features) * 100:.2f}%",
            'unknown': f"{(features['label'] == 0).sum() / len(features) * 100:.2f}%"
        }
    }

    return stats


def create_manual_labeling_candidates(features, top_k=50):
    """
    Identify top candidates for manual labeling
    These are important nodes that are currently unlabeled
    """
    print(f"\nIdentifying top {top_k} candidates for manual labeling...")

    unlabeled = features[features['label'] == 0].copy()

    # Score nodes by importance for labeling
    # Prioritize: high degree, high pagerank, high value
    unlabeled['labeling_priority'] = (
            unlabeled['total_degree'] / unlabeled['total_degree'].max() * 0.4 +
            unlabeled['pagerank'] / unlabeled['pagerank'].max() * 0.3 +
            (unlabeled['total_received'] + unlabeled['total_sent']) /
            (unlabeled['total_received'] + unlabeled['total_sent']).max() * 0.3
    )

    candidates = unlabeled.nlargest(top_k, 'labeling_priority')[
        ['node_id', 'address', 'total_degree', 'pagerank', 'total_received',
         'total_sent', 'node_type', 'labeling_priority']
    ]

    # Save candidates to CSV for manual review
    candidates.to_csv('manual_labeling_candidates.csv', index=False)
    print(f"  ✓ Saved top candidates to: manual_labeling_candidates.csv")
    print(f"  → Review these addresses and add them to blacklist/whitelist")

    return candidates


def save_labeled_data(features, stats):
    """
    Save labeled data and statistics
    """
    print("\nSaving labeled data...")

    # Save labeled nodes
    features.to_csv(OUTPUT_LABELED, index=False)
    print(f"  ✓ Labeled nodes saved to: {OUTPUT_LABELED}")

    # Save statistics
    with open(OUTPUT_LABEL_STATS, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Label statistics saved to: {OUTPUT_LABEL_STATS}")

    print(f"\n{'=' * 60}")
    print("LABEL ASSIGNMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Labeled nodes: {stats['labeled_nodes']} ({stats['labeled_nodes'] / stats['total_nodes'] * 100:.1f}%)")
    print(f"  - Illicit: {stats['illicit_nodes']} ({stats['label_percentages']['illicit']})")
    print(f"  - Licit: {stats['licit_nodes']} ({stats['label_percentages']['licit']})")
    print(f"Unlabeled nodes: {stats['unlabeled_nodes']} ({stats['label_percentages']['unknown']})")
    print(f"\nConfidence Distribution:")
    print(f"  - High (>0.8): {stats['confidence_distribution']['high (>0.8)']}")
    print(f"  - Medium (0.5-0.8): {stats['confidence_distribution']['medium (0.5-0.8)']}")
    print(f"  - Low (<0.5): {stats['confidence_distribution']['low (<0.5)']}")
    print(f"{'=' * 60}\n")


def main():
    """
    Main function to orchestrate label assignment
    """
    print("\n" + "=" * 60)
    print("LABEL ASSIGNMENT - PHASE 5")
    print("=" * 60)

    # Load features
    features = load_features()
    edges = load_edges()

    # Assign known labels from blacklist/whitelist
    features = assign_known_labels(features)

    # Apply heuristic rules
    features = apply_heuristic_suspicious_patterns(features, edges)
    features = apply_heuristic_legitimate_patterns(features, edges)

    # Generate statistics
    stats = generate_label_statistics(features)

    # Create manual labeling candidates
    candidates = create_manual_labeling_candidates(features)

    # Save results
    save_labeled_data(features, stats)

    print("✓ Label assignment completed!")
    print("\n📝 NEXT STEPS:")
    print("1. Review 'manual_labeling_candidates.csv'")
    print("2. Research high-priority addresses using:")
    print("   - Etherscan.io (check transaction patterns)")
    print("   - Public blacklists (Chainalysis, etc.)")
    print("   - Community reports")
    print("3. Add confirmed addresses to KNOWN_ILLICIT/LEGITIMATE lists")
    print("4. Re-run this script to update labels")
    print("\n  Ready for Phase 6: Data Preparation for GNN Training")


if __name__ == "__main__":
    main()