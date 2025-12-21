# HERO: Hierarchical Equitable Routing Optimization

HERO (Hierarchical Equitable Routing Optimization) is a novel metaheuristic for solving fairness-aware Vehicle Routing Problems (VRP) with Time Windows. The method employs Hierarchical Navigable Small World (HNSW) graphs to accelerate candidate generation in Adaptive Large Neighborhood Search (ALNS), achieving 1.48× speedup over classical ALNS (70.03s vs. 103.38s average runtime) with speedup increasing to 2.62× for large instances (1000 customers). The HNSW-accelerated variant (ALNS-HNSW) alone provides 1.36× speedup overall (1.99× for large instances). HERO achieves significant fairness improvements (15.20% CV reduction vs. ALNS, 18.42% vs. ALNS-HNSW) while being 2.37-2.54× faster than exact solvers (OR-Tools, PyVRP). The method demonstrates exceptional reliability with 100% HNSW query success rate across 4.45 million+ queries with zero failures, and supports real-time dynamic updates through incremental HNSW index modifications (averaging 88.3% incremental update ratio).

## Features

- **HNSW-Accelerated ALNS**: Reduces candidate generation complexity from O(n²) to O(k log n)
- **Fairness-Aware Optimization**: Simultaneously optimizes cost, driver workload equity (CV), and route workload distribution fairness (Jain's index)
- **14-Dimensional Feature Encoding**: Incorporates spatial, temporal, capacity, fairness gradients, and subsequence-aware features
- **Dynamic VRP Support**: Real-time handling of new customer arrivals with incremental HNSW index updates
- **Comprehensive Benchmarks**: Evaluation on Solomon, Homberger, CVRP X, and Euro-NeurIPS 2022 instances
- **Baseline Comparisons**: Includes comparisons with OR-Tools and PyVRP

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/asimsinan/hero.git
cd hero
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import faiss; import numpy; print('Installation successful')"
```

## Quick Start

### Running a Single Instance

Solve a single VRP instance using HERO:

```bash
python -m src.experiments.run_experiment \
    --instance data/benchmarks/solomon_original/C101.TXT \
    --iterations 200 \
    --seed 42 \
    --alpha 1.0 \
    --beta 0.3 \
    --gamma 0.2
```

### Running Quick Benchmark

Test the system with a small synthetic instance:

```bash
python -m src.experiments.run_experiment --quick --iterations 100
```

### Running Comprehensive Benchmarks

Run full benchmark suite across all instance types:

```bash
python scripts/comprehensive_benchmark.py \
    --seeds 42 123 456 \
    --max-iterations 200 \
    --output-dir results/benchmark
```

For a quick test (fewer instances):

```bash
python scripts/comprehensive_benchmark.py --quick
```

## Replicating Paper Results

### Step 1: Prepare Benchmark Data

The benchmark instances should already be included in the repository under `hnsw_fairvrp/data/benchmarks/`. If missing, the instances are organized as follows:

- `solomon_original/`: Solomon VRPTW instances (100 customers)
- `homberger/`: Homberger extended instances (200-1000 customers)
- `cvrp/`: CVRP X instances (101-1001 customers)
- `euro_neurips_2022/`: Euro-NeurIPS 2022 real-world instances (200-880 customers)

### Step 2: Run Experiments

To replicate the main results from the paper, run the comprehensive benchmark with the same configuration:

```bash
cd hnsw_fairvrp
python scripts/comprehensive_benchmark.py \
    --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 \
    --max-iterations 1500 \
    --methods alns alns_hnsw hero ortools pyvrp \
    --output-dir results/paper_replication
```

This will:
- Run all methods (ALNS, ALNS-HNSW, HERO, OR-Tools, PyVRP) on all benchmark instances
- Use 10 random seeds for statistical significance (matching paper experiments)
- Save results to `results/paper_replication/`

### Step 3: Analyze Results

Results are saved in multiple formats:

- **JSON files**: Detailed results per instance/method/seed in `results/paper_replication/detailed/`
- **CSV files**: Summary tables in `results/paper_replication/summary/`
- **Log files**: Execution logs with progress tracking

To analyze results:

```bash
python -m src.experiments.analysis \
    --results-dir results/paper_replication \
    --output analysis_report.html
```

### Step 4: Generate Tables and Figures

The paper's figures can be generated from the results:

```bash
# Generate all figures
python scripts/generate_figures.py \
    --results-dir results/paper_replication \
    --output-dir figures/paper

# Generate specific figures only
python scripts/generate_figures.py \
    --results-dir results/paper_replication \
    --output-dir figures/paper \
    --figures speedup scalability fairness pareto

# Available figures:
# - speedup: Speedup vs Instance Size
# - scalability: Scalability Analysis (log-log)
# - fairness: Fairness Improvement Bar Chart
# - pareto: Cost-Fairness Trade-off (Pareto Fronts)
# - ablation: Ablation Study Results
# - convergence: Convergence Curves
```

The script generates both PDF (for LaTeX) and PNG (for preview) versions of each figure.

## Configuration

### ALNS Parameters

Key parameters for the ALNS solver:

- `--iterations`: Maximum number of ALNS iterations (default: 100)
- `--alpha`: Weight for cost objective (default: 1.0)
- `--beta`: Weight for driver fairness (CV) (default: 0.3)
- `--gamma`: Weight for route workload distribution fairness (Jain's index) (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

### HNSW Parameters

HNSW-specific parameters (set via environment or config):

- `HNSW_M`: Number of bi-directional links per node (default: 16)
- `HNSW_EF_CONSTRUCTION`: Construction-time search width (default: 200)
- `HNSW_EF_SEARCH`: Query-time search width (default: 50)
- `HNSW_K`: Number of candidates to retrieve (default: adaptive based on instance size)

### Example: Custom Configuration

```bash
python -m src.experiments.run_experiment \
    --instance data/benchmarks/solomon_original/C101.TXT \
    --iterations 500 \
    --alpha 1.0 \
    --beta 0.5 \
    --gamma 0.3 \
    --seed 42
```

## Method Comparison

The benchmark script supports multiple methods:

- `alns`: Classical ALNS without HNSW acceleration
- `alns_hnsw`: HNSW-accelerated ALNS without fairness features (1.36× speedup overall, 1.99× for large instances)
- `hero`: Full HERO with HNSW acceleration and fairness-aware objective (1.48× speedup overall, 2.62× for large instances, 15-18% CV reduction)
- `ortools`: Google OR-Tools solver (requires installation)
- `pyvrp`: PyVRP Hybrid Genetic Search (requires installation)

To run only HERO variants:

```bash
python scripts/comprehensive_benchmark.py \
    --methods alns alns_hnsw hero \
    --no-external
```

## Ablation Studies

Run ablation studies to analyze component contributions:

```bash
python scripts/ablation_study.py \
    --instance data/benchmarks/solomon_original/C101.TXT \
    --seeds 42 123 456 \
    --output-dir results/ablation
```

This evaluates:
- HNSW acceleration contribution
- Fairness feature contribution
- Subsequence-aware feature contribution
- Adaptive k selection impact

## Key Components

### Feature Encoder (`src/hnsw/features.py`)

Encodes VRP customers and insertion positions into 14-dimensional feature vectors:
- Spatial features (2D): Customer coordinates
- Temporal features (3D): Time windows, service time
- Capacity features (2D): Demand, route load
- Fairness features (3D): CV gradient, Jain gradient, route imbalance
- Subsequence features (3D): Prefix/suffix slack, combined slack
- Route features (1D): Route length

### HNSW Manager (`src/hnsw/manager.py`)

Manages the HNSW index for approximate nearest neighbor search:
- Index initialization and updates
- Candidate retrieval with adaptive k
- Incremental updates for dynamic scenarios

### ALNS Solver (`src/heuristics/alns.py`)

Adaptive Large Neighborhood Search framework:
- Destroy operators: Random, Related, Worst, Route removal
- Repair operators: HNSW-guided greedy and regret insertion
- Simulated annealing acceptance criterion
- Adaptive operator weight adjustment

## Performance Results

Based on comprehensive experiments across standard VRPTW benchmarks (Solomon, Homberger, CVRP, Euro-NeurIPS 2022) with 1,500 successful runs (10 seeds):

- **HERO**: 1.48× speedup over ALNS (70.03s vs. 103.38s), increasing to 2.62× for large instances (1000 customers)
- **ALNS-HNSW**: 1.36× speedup over ALNS (75.74s vs. 103.38s), increasing to 1.99× for large instances
- **Fairness Improvements**: 15.20% CV reduction vs. ALNS, 18.42% vs. ALNS-HNSW
- **HNSW Reliability**: 100% query success rate across 4.45 million+ queries (2.41M for ALNS-HNSW, 2.04M for HERO) with zero failures
- **Incremental Updates**: Averages 88.3% incremental update ratio (89.7% of measurements achieve 80%+) for dynamic scenarios
- **Comparison with Exact Solvers**: HERO is 2.37× faster than OR-Tools and 2.54× faster than PyVRP while enabling fairness optimization

## Performance Tips

1. **For Large Instances**: Increase `--max-iterations` for better solutions (paper uses 1500 iterations)
2. **For Speed**: Use `--no-external` to skip OR-Tools and PyVRP comparisons
3. **For Reproducibility**: Always set `--seed` for deterministic results
4. **For Memory**: HNSW index uses O(n·d) memory where n is number of positions and d=14

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running from the project root:

```bash
cd hero/hnsw_fairvrp
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### HNSW Library Issues

If `faiss-cpu` fails to install:

```bash
pip install --upgrade pip
pip install faiss-cpu --no-cache-dir
```

For GPU support (optional), use `faiss-gpu` instead:
```bash
pip install faiss-gpu --no-cache-dir
```

### Memory Issues

For very large instances (>5000 customers), consider:
- Reducing `HNSW_EF_CONSTRUCTION` and `HNSW_EF_SEARCH`
- Using smaller `HNSW_K` values
- Running with fewer iterations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact:
- Email: asimyuksel@sdu.edu.tr

