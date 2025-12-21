#!/usr/bin/env python3
"""Analyze HNSW query failures from log files.

This script analyzes log files to identify patterns in HNSW query failures
and provides recommendations for parameter tuning.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import glob

def parse_log_file(log_path: Path) -> Dict:
    """Parse log file and extract HNSW failure information.
    
    FIXED: Tracks queries and failures per stats line to calculate accurate failure rates.
    Only counts failures that occur between consecutive stats lines to avoid double-counting.
    """
    failures = []
    successes = []
    
    # FIXED: Track queries and failures per stats line (per run)
    # Each stats line represents one run, count failures only for that run
    stats_lines = []  # Track (line_index, method, queries) tuples
    failures_by_stats = defaultdict(int)  # Track failures per stats line index
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # First pass: Find all stats lines and extract method/queries/failures
    for i, line in enumerate(lines):
        if 'HNSW stats:' in line:
            method_match = re.search(r'(\w+)\s+HNSW stats:', line)
            queries_match = re.search(r'queries=(\d+)', line)
            # FIXED: Extract failures from stats line if available
            failures_match = re.search(r'failures=(\d+)', line)
            if method_match and queries_match:
                method = method_match.group(1)
                queries = int(queries_match.group(1))
                failures_from_stats = int(failures_match.group(1)) if failures_match else 0
                # FIXED: Store as tuple with 4 elements: (line_idx, method, queries, failures)
                stats_lines.append((i, method, queries, failures_from_stats))
                successes.append(queries)
                # FIXED: Use failures from stats if available (most accurate)
                if failures_from_stats > 0:
                    failures_by_stats[i] = failures_from_stats
    
    # Second pass: Count failures between consecutive stats lines
    # Each failure is counted only for the stats line immediately before it
    # This ensures we don't double-count failures across runs
    # NOTE: If failures are already in stats line, we skip this counting
    for i, line in enumerate(lines):
        if 'HNSW query failed for customer' in line:
            # Find the most recent stats line before this failure
            # Only count if there's a stats line before it AND it doesn't already have failures
            for stats_data in reversed(stats_lines):
                if len(stats_data) == 4:  # New format with failures
                    stats_idx, method, queries, failures_from_stats = stats_data
                else:  # Old format
                    stats_idx, method, queries = stats_data
                    failures_from_stats = 0
                
                if stats_idx < i:
                    # Only count if failures weren't already in stats line
                    if failures_from_stats == 0:
                        failures_by_stats[stats_idx] = failures_by_stats.get(stats_idx, 0) + 1
                    break
            
            # Extract failure info for analysis
            customer_match = re.search(r'customer (\d+)', line)
            customer_id = int(customer_match.group(1)) if customer_match else None
            
            error_type_match = re.search(r': (\w+) -', line)
            error_type = error_type_match.group(1) if error_type_match else 'Unknown'
            
            k_match = re.search(r'k=(\d+)', line)
            ef_match = re.search(r'ef=(\d+)', line)
            size_match = re.search(r'size=(\d+)', line)
            
            failure_info = {
                'line': i + 1,
                'customer_id': customer_id,
                'error_type': error_type,
                'k': int(k_match.group(1)) if k_match else None,
                'ef': int(ef_match.group(1)) if ef_match else None,
                'size': int(size_match.group(1)) if size_match else None,
                'raw_line': line.strip()
            }
            failures.append(failure_info)
    
    # FIXED: Aggregate failures per method from stats lines
    queries_by_method = defaultdict(int)
    failures_by_method = defaultdict(int)
    for stats_line_data in stats_lines:
        if len(stats_line_data) == 4:  # New format with failures
            stats_idx, method, queries, failures_from_stats = stats_line_data
            queries_by_method[method] += queries
            # FIXED: Prefer failures from stats line (most accurate), otherwise use counted failures
            if failures_from_stats > 0:
                failures_by_method[method] += failures_from_stats
            else:
                failures_by_method[method] += failures_by_stats[stats_idx]
        else:  # Old format without failures
            stats_idx, method, queries = stats_line_data
            queries_by_method[method] += queries
            failures_by_method[method] += failures_by_stats[stats_idx]
    
    # FIXED: Calculate accurate failure rates
    total_queries = sum(queries_by_method.values())
    total_failures = sum(failures_by_method.values())
    failure_rate = (total_failures / total_queries * 100) if total_queries > 0 else 0.0
    success_rate = ((total_queries - total_failures) / total_queries * 100) if total_queries > 0 else 0.0
    
    return {
        'failures': failures,
        'total_successes': sum(successes),
        'total_failures': len(failures),
        # FIXED: Add accurate statistics
        'total_queries': total_queries,
        'failure_rate': failure_rate,
        'success_rate': success_rate,
        'queries_by_method': dict(queries_by_method),
        'failures_by_method': dict(failures_by_method),
    }

def analyze_failures(data: Dict) -> Dict:
    """Analyze failure patterns and provide insights."""
    failures = data['failures']
    
    if not failures:
        return {'message': 'No failures found'}
    
    # Group by error type
    error_types = defaultdict(int)
    for f in failures:
        error_types[f['error_type']] += 1
    
    # Group by index size ranges
    size_ranges = {
        'very_small (<30)': 0,
        'small (30-50)': 0,
        'medium (50-100)': 0,
        'large (100-500)': 0,
        'very_large (>500)': 0,
        'unknown': 0
    }
    
    for f in failures:
        size = f['size']
        if size is None:
            size_ranges['unknown'] += 1
        elif size < 30:
            size_ranges['very_small (<30)'] += 1
        elif size < 50:
            size_ranges['small (30-50)'] += 1
        elif size < 100:
            size_ranges['medium (50-100)'] += 1
        elif size < 500:
            size_ranges['large (100-500)'] += 1
        else:
            size_ranges['very_large (>500)'] += 1
    
    # Analyze k values
    k_values = [f['k'] for f in failures if f['k'] is not None]
    k_stats = {
        'min': min(k_values) if k_values else None,
        'max': max(k_values) if k_values else None,
        'avg': sum(k_values) / len(k_values) if k_values else None,
    }
    
    # Analyze ef values
    ef_values = [f['ef'] for f in failures if f['ef'] is not None]
    ef_stats = {
        'min': min(ef_values) if ef_values else None,
        'max': max(ef_values) if ef_values else None,
        'avg': sum(ef_values) / len(ef_values) if ef_values else None,
    }
    
    # Analyze size values
    sizes = [f['size'] for f in failures if f['size'] is not None]
    size_stats = {
        'min': min(sizes) if sizes else None,
        'max': max(sizes) if sizes else None,
        'avg': sum(sizes) / len(sizes) if sizes else None,
    }
    
    # Find problematic customers
    customer_failures = defaultdict(int)
    for f in failures:
        if f['customer_id'] is not None:
            customer_failures[f['customer_id']] += 1
    
    top_failing_customers = sorted(
        customer_failures.items(), 
        key=lambda x: -x[1]
    )[:10]
    
    return {
        'total_failures': len(failures),
        'error_types': dict(error_types),
        'size_distribution': size_ranges,
        'k_stats': k_stats,
        'ef_stats': ef_stats,
        'size_stats': size_stats,
        'top_failing_customers': top_failing_customers,
    }

def generate_recommendations(analysis: Dict) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    if not analysis or 'message' in analysis:
        return ['No failures to analyze']
    
    total = analysis['total_failures']
    
    # Check size distribution
    size_dist = analysis['size_distribution']
    if size_dist['very_small (<30)'] > total * 0.5:
        recommendations.append(
            f"⚠️  {size_dist['very_small (<30)']} failures on very small indices (<30). "
            "These are already handled by skipping HNSW. Consider if this is expected."
        )
    
    if size_dist['small (30-50)'] > total * 0.3:
        recommendations.append(
            f"⚠️  {size_dist['small (30-50)']} failures on small indices (30-50). "
            "Consider: Lower k for small indices, increase ef_search, or skip HNSW for indices <50."
        )
    
    # Check error types
    error_types = analysis['error_types']
    if 'RuntimeError' in error_types and error_types['RuntimeError'] > total * 0.5:
        recommendations.append(
            f"⚠️  {error_types['RuntimeError']} RuntimeErrors. "
            "Likely 'Cannot return results' errors. Consider: Increase M, ef_construction, or ef_search."
        )
    
    # Check k values
    k_stats = analysis['k_stats']
    if k_stats['avg'] and k_stats['avg'] > 15:
        recommendations.append(
            f"⚠️  Average k={k_stats['avg']:.1f} is high. "
            "Consider: More aggressive adaptive k reduction for small indices."
        )
    
    # Check ef values
    ef_stats = analysis['ef_stats']
    if ef_stats['avg'] and ef_stats['avg'] < 150:
        recommendations.append(
            f"⚠️  Average ef={ef_stats['avg']:.1f} may be too low. "
            "Consider: Increase base ef_search or make ef_search calculation more aggressive."
        )
    
    # Check size stats
    size_stats = analysis['size_stats']
    if size_stats['avg'] and size_stats['avg'] < 100:
        recommendations.append(
            f"⚠️  Average index size={size_stats['avg']:.0f} is small. "
            "Consider: Skip HNSW for indices <50 or use more conservative parameters."
        )
    
    if not recommendations:
        recommendations.append("✅ No obvious patterns detected. Failures may be random edge cases.")
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Analyze HNSW query failures')
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Path to log file to analyze'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('hnsw_fairvrp'),
        help='Directory to search for log files'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Analyze the most recent log file'
    )
    args = parser.parse_args()
    
    # Find log file
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent
            log_path = script_dir / args.log_file
    elif args.latest or True:  # Default to latest
        # Search in multiple locations
        search_dirs = [
            Path(args.log_dir),
            Path(__file__).parent.parent,  # hnsw_fairvrp/
            Path(__file__).parent.parent.parent,  # HNSW-VRP/
        ]
        log_files = []
        for search_dir in search_dirs:
            log_files.extend(list(search_dir.glob('benchmark_run_*.log')))
            log_files.extend(list(search_dir.glob('**/benchmark_run_*.log')))
        
        if not log_files:
            print("No log files found")
            return
        log_path = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"Analyzing latest log: {log_path}")
    
    # Parse and analyze
    print("=" * 80)
    print("HNSW FAILURE ANALYSIS")
    print("=" * 80)
    print()
    
    data = parse_log_file(log_path)
    analysis = analyze_failures(data)
    
    # Print results
    print(f"Total failures: {analysis.get('total_failures', 0)}")
    print(f"Total queries: {data.get('total_queries', 0)}")
    print(f"Total successes: {data.get('total_successes', 0)}")
    
    # FIXED: Use accurate failure rate from data
    if 'failure_rate' in data:
        print(f"Failure rate: {data['failure_rate']:.1f}%")
        print(f"Success rate: {data['success_rate']:.1f}%")
    elif data.get('total_queries', 0) > 0:
        failure_rate = 100 * analysis.get('total_failures', 0) / data.get('total_queries', 1)
        success_rate = 100 * (data.get('total_queries', 0) - analysis.get('total_failures', 0)) / data.get('total_queries', 1)
        print(f"Failure rate: {failure_rate:.1f}%")
        print(f"Success rate: {success_rate:.1f}%")
    print()
    
    # FIXED: Print per-method breakdown
    if 'queries_by_method' in data and 'failures_by_method' in data:
        print("PER-METHOD BREAKDOWN:")
        for method in sorted(data['queries_by_method'].keys()):
            queries = data['queries_by_method'][method]
            failures = data['failures_by_method'].get(method, 0)
            if queries > 0:
                method_failure_rate = 100 * failures / queries
                print(f"  {method}: {failures}/{queries} failures ({method_failure_rate:.1f}%)")
        print()
    
    if 'error_types' in analysis:
        print("ERROR TYPES:")
        for error_type, count in sorted(analysis['error_types'].items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        print()
    
    if 'size_distribution' in analysis:
        print("FAILURES BY INDEX SIZE:")
        for size_range, count in analysis['size_distribution'].items():
            if count > 0:
                print(f"  {size_range}: {count}")
        print()
    
    if 'k_stats' in analysis and analysis['k_stats']['avg']:
        print("K STATISTICS:")
        print(f"  Min: {analysis['k_stats']['min']}, Max: {analysis['k_stats']['max']}, Avg: {analysis['k_stats']['avg']:.1f}")
        print()
    
    if 'ef_stats' in analysis and analysis['ef_stats']['avg']:
        print("EF STATISTICS:")
        print(f"  Min: {analysis['ef_stats']['min']}, Max: {analysis['ef_stats']['max']}, Avg: {analysis['ef_stats']['avg']:.1f}")
        print()
    
    if 'size_stats' in analysis and analysis['size_stats']['avg']:
        print("INDEX SIZE STATISTICS:")
        print(f"  Min: {analysis['size_stats']['min']}, Max: {analysis['size_stats']['max']}, Avg: {analysis['size_stats']['avg']:.0f}")
        print()
    
    if 'top_failing_customers' in analysis:
        print("TOP FAILING CUSTOMERS:")
        for customer_id, count in analysis['top_failing_customers']:
            print(f"  Customer {customer_id}: {count} failures")
        print()
    
    # Recommendations
    recommendations = generate_recommendations(analysis)
    print("RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  {rec}")
    print()

if __name__ == '__main__':
    main()

