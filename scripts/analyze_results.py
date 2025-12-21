#!/usr/bin/env python3
"""
Comprehensive analysis of HERO experimental results.

Analyzes results and provides insights on:
- Speedup analysis
- Fairness improvements
- Cost trade-offs
- Statistical significance
- Comparison with baselines
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results from JSON or CSV."""
    results_dir = Path(results_dir)
    
    # Try JSON first
    json_files = list(results_dir.glob("**/*results*.json"))
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        return df
    
    # Try CSV
    csv_files = list(results_dir.glob("**/*results*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        return df
    
    raise ValueError(f"No results found in {results_dir}")


def analyze_speedup(df: pd.DataFrame) -> Dict:
    """Analyze speedup of HNSW and HERO over ALNS."""
    results = {}
    
    # Filter valid results (non-inf, non-zero time)
    valid = df[(df['time_seconds'] > 0) & (df['cost'] != float('inf'))].copy()
    
    if len(valid) == 0:
        return {"error": "No valid results found"}
    
    # Group by method and instance size
    by_size_method = valid.groupby(['n_customers', 'method'])['time_seconds'].agg(['mean', 'std', 'count'])
    
    # Calculate speedup for each instance size
    speedups = {}
    for size in sorted(valid['n_customers'].unique()):
        size_data = valid[valid['n_customers'] == size]
        
        alns_times = size_data[size_data['method'] == 'alns']['time_seconds']
        hnsw_times = size_data[size_data['method'] == 'alns_hnsw']['time_seconds']
        hero_times = size_data[size_data['method'] == 'hero']['time_seconds']
        
        if len(alns_times) > 0:
            alns_mean = alns_times.mean()
            
            if len(hnsw_times) > 0:
                hnsw_mean = hnsw_times.mean()
                speedups[f'n={size}_hnsw'] = {
                    'speedup': alns_mean / hnsw_mean if hnsw_mean > 0 else 0,
                    'alns_time': alns_mean,
                    'hnsw_time': hnsw_mean,
                }
            
            if len(hero_times) > 0:
                hero_mean = hero_times.mean()
                speedups[f'n={size}_hero'] = {
                    'speedup': alns_mean / hero_mean if hero_mean > 0 else 0,
                    'alns_time': alns_mean,
                    'hero_time': hero_mean,
                }
    
    # Overall speedup
    alns_all = valid[valid['method'] == 'alns']['time_seconds']
    hnsw_all = valid[valid['method'] == 'alns_hnsw']['time_seconds']
    hero_all = valid[valid['method'] == 'hero']['time_seconds']
    
    results['by_size'] = speedups
    results['overall'] = {
        'hnsw_speedup': alns_all.mean() / hnsw_all.mean() if len(hnsw_all) > 0 and hnsw_all.mean() > 0 else 0,
        'hero_speedup': alns_all.mean() / hero_all.mean() if len(hero_all) > 0 and hero_all.mean() > 0 else 0,
        'alns_mean_time': alns_all.mean(),
        'hnsw_mean_time': hnsw_all.mean(),
        'hero_mean_time': hero_all.mean(),
    }
    
    return results


def analyze_fairness(df: pd.DataFrame) -> Dict:
    """Analyze fairness improvements."""
    results = {}
    
    # Filter valid results
    valid = df[(df['cv'] >= 0) & (df['cost'] != float('inf'))].copy()
    
    # Group by method
    by_method = valid.groupby('method')['cv'].agg(['mean', 'std', 'count'])
    
    alns_cv = valid[valid['method'] == 'alns']['cv']
    hnsw_cv = valid[valid['method'] == 'alns_hnsw']['cv']
    hero_cv = valid[valid['method'] == 'hero']['cv']
    
    results['by_method'] = {
        'alns': {'mean': alns_cv.mean(), 'std': alns_cv.std()} if len(alns_cv) > 0 else None,
        'alns_hnsw': {'mean': hnsw_cv.mean(), 'std': hnsw_cv.std()} if len(hnsw_cv) > 0 else None,
        'hero': {'mean': hero_cv.mean(), 'std': hero_cv.std()} if len(hero_cv) > 0 else None,
    }
    
    # Calculate improvements
    if len(alns_cv) > 0 and len(hero_cv) > 0:
        alns_mean = alns_cv.mean()
        hero_mean = hero_cv.mean()
        improvement_vs_alns = ((alns_mean - hero_mean) / alns_mean * 100) if alns_mean > 0 else 0
        results['improvement_vs_alns'] = improvement_vs_alns
    
    if len(hnsw_cv) > 0 and len(hero_cv) > 0:
        hnsw_mean = hnsw_cv.mean()
        hero_mean = hero_cv.mean()
        improvement_vs_hnsw = ((hnsw_mean - hero_mean) / hnsw_mean * 100) if hnsw_mean > 0 else 0
        results['improvement_vs_hnsw'] = improvement_vs_hnsw
    
    return results


def analyze_cost_tradeoff(df: pd.DataFrame) -> Dict:
    """Analyze cost-fairness trade-off."""
    results = {}
    
    # Filter valid results
    valid = df[(df['cost'] > 0) & (df['cost'] != float('inf')) & (df['cv'] >= 0)].copy()
    
    # Group by method
    alns = valid[valid['method'] == 'alns']
    hnsw = valid[valid['method'] == 'alns_hnsw']
    hero = valid[valid['method'] == 'hero']
    
    results['by_method'] = {
        'alns': {
            'mean_cost': alns['cost'].mean() if len(alns) > 0 else 0,
            'mean_cv': alns['cv'].mean() if len(alns) > 0 else 0,
        },
        'alns_hnsw': {
            'mean_cost': hnsw['cost'].mean() if len(hnsw) > 0 else 0,
            'mean_cv': hnsw['cv'].mean() if len(hnsw) > 0 else 0,
        },
        'hero': {
            'mean_cost': hero['cost'].mean() if len(hero) > 0 else 0,
            'mean_cv': hero['cv'].mean() if len(hero) > 0 else 0,
        },
    }
    
    # Calculate cost increase for fairness
    if len(hnsw) > 0 and len(hero) > 0:
        hnsw_cost = hnsw['cost'].mean()
        hero_cost = hero['cost'].mean()
        cost_increase = ((hero_cost - hnsw_cost) / hnsw_cost * 100) if hnsw_cost > 0 else 0
        results['cost_increase_vs_hnsw'] = cost_increase
    
    if len(alns) > 0 and len(hero) > 0:
        alns_cost = alns['cost'].mean()
        hero_cost = hero['cost'].mean()
        cost_change = ((hero_cost - alns_cost) / alns_cost * 100) if alns_cost > 0 else 0
        results['cost_change_vs_alns'] = cost_change
    
    return results


def statistical_tests(df: pd.DataFrame) -> Dict:
    """Perform statistical significance tests."""
    results = {}
    
    # Filter valid results
    valid = df[(df['cost'] > 0) & (df['cost'] != float('inf'))].copy()
    
    # Wilcoxon signed-rank test for paired samples (same instance, same seed)
    tests = []
    
    # Compare HNSW vs ALNS (time)
    alns_times = valid[valid['method'] == 'alns'].groupby(['instance_name', 'seed'])['time_seconds'].first()
    hnsw_times = valid[valid['method'] == 'alns_hnsw'].groupby(['instance_name', 'seed'])['time_seconds'].first()
    
    # Align by (instance, seed)
    common_keys = set(alns_times.index) & set(hnsw_times.index)
    if len(common_keys) > 0:
        alns_aligned = [alns_times[k] for k in common_keys]
        hnsw_aligned = [hnsw_times[k] for k in common_keys]
        
        if len(alns_aligned) > 0 and len(hnsw_aligned) > 0:
            stat, pvalue = stats.wilcoxon(alns_aligned, hnsw_aligned, alternative='two-sided')
            tests.append({
                'comparison': 'HNSW vs ALNS (time)',
                'statistic': stat,
                'pvalue': pvalue,
                'significant': pvalue < 0.05,
            })
    
    # Compare HERO vs HNSW (CV - fairness)
    hnsw_cvs = valid[valid['method'] == 'alns_hnsw'].groupby(['instance_name', 'seed'])['cv'].first()
    hero_cvs = valid[valid['method'] == 'hero'].groupby(['instance_name', 'seed'])['cv'].first()
    
    common_keys = set(hnsw_cvs.index) & set(hero_cvs.index)
    if len(common_keys) > 0:
        hnsw_aligned = [hnsw_cvs[k] for k in common_keys]
        hero_aligned = [hero_cvs[k] for k in common_keys]
        
        if len(hnsw_aligned) > 0 and len(hero_aligned) > 0:
            stat, pvalue = stats.wilcoxon(hnsw_aligned, hero_aligned, alternative='two-sided')
            tests.append({
                'comparison': 'HERO vs HNSW (CV)',
                'statistic': stat,
                'pvalue': pvalue,
                'significant': pvalue < 0.05,
            })
    
    results['wilcoxon_tests'] = tests
    return results


def print_analysis_report(df: pd.DataFrame, speedup: Dict, fairness: Dict, cost: Dict, stats_results: Dict):
    """Print comprehensive analysis report."""
    print("="*80)
    print("HERO EXPERIMENTAL RESULTS ANALYSIS")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Instances: {df['instance_name'].nunique()}")
    print(f"  Methods: {', '.join(df['method'].unique())}")
    print(f"  Seeds: {df['seed'].nunique()}")
    print(f"  Instance sizes: {sorted(df['n_customers'].unique())}")
    
    # Speedup Analysis
    print(f"\n{'='*80}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*80}")
    
    if 'overall' in speedup:
        overall = speedup['overall']
        print(f"\nOverall Speedup (across all instances):")
        print(f"  HNSW vs ALNS: {overall['hnsw_speedup']:.2f}×")
        print(f"    ALNS mean time: {overall['alns_mean_time']:.2f}s")
        print(f"    HNSW mean time: {overall['hnsw_mean_time']:.2f}s")
        print(f"  HERO vs ALNS: {overall['hero_speedup']:.2f}×")
        print(f"    HERO mean time: {overall['hero_mean_time']:.2f}s")
    
    if 'by_size' in speedup:
        print(f"\nSpeedup by Instance Size:")
        for key, data in sorted(speedup['by_size'].items()):
            print(f"  {key}: {data['speedup']:.2f}×")
    
    # Fairness Analysis
    print(f"\n{'='*80}")
    print("FAIRNESS ANALYSIS")
    print(f"{'='*80}")
    
    if 'by_method' in fairness:
        bm = fairness['by_method']
        print(f"\nAverage CV (Coefficient of Variation) by Method:")
        if bm.get('alns'):
            print(f"  ALNS:  {bm['alns']['mean']:.4f} ± {bm['alns']['std']:.4f}")
        if bm.get('alns_hnsw'):
            print(f"  ALNS-HNSW:  {bm['alns_hnsw']['mean']:.4f} ± {bm['alns_hnsw']['std']:.4f}")
        if bm.get('hero'):
            print(f"  HERO:  {bm['hero']['mean']:.4f} ± {bm['hero']['std']:.4f}")
    
    if 'improvement_vs_alns' in fairness:
        print(f"\nFairness Improvement:")
        print(f"  HERO vs ALNS: {fairness['improvement_vs_alns']:.2f}% CV reduction")
    
    if 'improvement_vs_hnsw' in fairness:
        print(f"  HERO vs HNSW: {fairness['improvement_vs_hnsw']:.2f}% CV reduction")
    
    # Cost Trade-off
    print(f"\n{'='*80}")
    print("COST-FAIRNESS TRADE-OFF")
    print(f"{'='*80}")
    
    if 'by_method' in cost:
        bm = cost['by_method']
        print(f"\nAverage Cost by Method:")
        if bm.get('alns'):
            print(f"  ALNS:  {bm['alns']['mean_cost']:.2f} (CV: {bm['alns']['mean_cv']:.4f})")
        if bm.get('alns_hnsw'):
            print(f"  ALNS-HNSW:  {bm['alns_hnsw']['mean_cost']:.2f} (CV: {bm['alns_hnsw']['mean_cv']:.4f})")
        if bm.get('hero'):
            print(f"  HERO:  {bm['hero']['mean_cost']:.2f} (CV: {bm['hero']['mean_cv']:.4f})")
    
    if 'cost_increase_vs_hnsw' in cost:
        print(f"\nCost Change for Fairness:")
        print(f"  HERO vs HNSW: {cost['cost_increase_vs_hnsw']:+.2f}% cost change")
    
    if 'cost_change_vs_alns' in cost:
        print(f"  HERO vs ALNS: {cost['cost_change_vs_alns']:+.2f}% cost change")
    
    # Statistical Tests
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*80}")
    
    if 'wilcoxon_tests' in stats_results:
        for test in stats_results['wilcoxon_tests']:
            sig = "YES" if test['significant'] else "NO"
            print(f"\n  {test['comparison']}:")
            print(f"    p-value: {test['pvalue']:.4f}")
            print(f"    Significant (p < 0.05): {sig}")
    
    # Baseline Status
    print(f"\n{'='*80}")
    print("BASELINE STATUS")
    print(f"{'='*80}")
    
    ortools = df[df['method'] == 'ortools']
    pyvrp = df[df['method'] == 'pyvrp']
    
    ortools_success = len(ortools[ortools['cost'] != float('inf')])
    pyvrp_success = len(pyvrp[pyvrp['cost'] != float('inf')])
    
    print(f"\n  OR-Tools: {ortools_success}/{len(ortools)} successful runs")
    print(f"  PyVRP: {pyvrp_success}/{len(pyvrp)} successful runs")
    
    if ortools_success == 0:
        print(f"    WARNING: OR-Tools failed on all instances")
    if pyvrp_success == 0:
        print(f"    WARNING: PyVRP failed on all instances")
    
    # Key Findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    findings = []
    
    if 'overall' in speedup and speedup['overall']['hnsw_speedup'] >= 1.5:
        findings.append(f"✓ HNSW achieves {speedup['overall']['hnsw_speedup']:.2f}× speedup (target: 2×)")
    elif 'overall' in speedup:
        findings.append(f"⚠ HNSW achieves {speedup['overall']['hnsw_speedup']:.2f}× speedup (target: 2×)")
    
    if 'improvement_vs_hnsw' in fairness and fairness['improvement_vs_hnsw'] > 0:
        findings.append(f"✓ HERO improves fairness by {fairness['improvement_vs_hnsw']:.2f}% vs HNSW")
    
    if 'cost_change_vs_alns' in cost and cost['cost_change_vs_alns'] < 10:
        findings.append(f"✓ HERO cost increase vs ALNS: {cost['cost_change_vs_alns']:.2f}% (acceptable)")
    elif 'cost_change_vs_alns' in cost:
        findings.append(f"⚠ HERO cost increase vs ALNS: {cost['cost_change_vs_alns']:.2f}% (high)")
    
    if 'cost_change_vs_alns' in cost and cost['cost_change_vs_alns'] < 0:
        findings.append(f"✓ HERO achieves BETTER cost than ALNS ({abs(cost['cost_change_vs_alns']):.2f}% reduction)")
    
    for finding in findings:
        print(f"  {finding}")
    
    print(f"\n{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze HERO experimental results")
    parser.add_argument('--results-dir', type=str, 
                       default='results/solomon_benchmark',
                       help='Directory containing results')
    parser.add_argument('--output', type=str,
                       help='Output file for analysis (JSON)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    df = load_results(Path(args.results_dir))
    print(f"Loaded {len(df)} results\n")
    
    # Perform analyses
    print("Analyzing speedup...")
    speedup = analyze_speedup(df)
    
    print("Analyzing fairness...")
    fairness = analyze_fairness(df)
    
    print("Analyzing cost trade-off...")
    cost = analyze_cost_tradeoff(df)
    
    print("Performing statistical tests...")
    stats_results = statistical_tests(df)
    
    # Print report
    print_analysis_report(df, speedup, fairness, cost, stats_results)
    
    # Save if requested
    if args.output:
        analysis = {
            'speedup': speedup,
            'fairness': fairness,
            'cost': cost,
            'statistical_tests': stats_results,
        }
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {args.output}")


if __name__ == '__main__':
    main()

