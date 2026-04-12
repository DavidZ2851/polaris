import os
import sys
import pandas as pd
from pathlib import Path

def analyze_results(base_dir, stages=3):
    results = []
    
    for folder in sorted(Path(base_dir).iterdir()):
        if not folder.is_dir():
            continue
        
        csv_path = folder / "eval_results.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        
        num_episodes = len(df)
        
        # Calculate stage success rates
        stage_rates = []
        for i in range(1, stages + 1):
            threshold = i / stages
            rate = (df['progress'] >= threshold - 0.01).mean() * 100
            stage_rates.append(rate)
        
        avg_rate = sum(stage_rates) / stages
        
        results.append({
            'run': folder.name,
            'episodes': num_episodes,
            'stages': stage_rates,
            'avg': avg_rate
        })
    
    # Print header
    stage_headers = " ".join([f"{'Stage'+str(i):>10}" for i in range(1, stages + 1)])
    print("=" * (50 + 10 + 11 * stages + 11))
    print(f"{'Run':<50} {'Episodes':>8} {stage_headers} {'Average':>10}")
    print("=" * (50 + 10 + 11 * stages + 11))
    
    # Print each run
    for row in results:
        stage_vals = " ".join([f"{s:>9.1f}%" for s in row['stages']])
        print(f"{row['run']:<50} {row['episodes']:>8} {stage_vals} {row['avg']:>9.1f}%")
    
    print("=" * (50 + 10 + 11 * stages + 11))

if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    stages = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    analyze_results(base_dir, stages)