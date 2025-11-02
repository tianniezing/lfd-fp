#!/usr/bin/env python
import re
from pathlib import Path

def parse_summary_log(file_path: Path) -> dict:
    """Parses a summary.log file to extract config and metrics."""
    content = file_path.read_text()
    try:
        config = {
            'run_id': int(re.search(r"RUN_ID: (\d+)", content).group(1)),
            'model': re.search(r"MODEL: ([\w/-]+)", content).group(1),
            'lr': float(re.search(r"LEARNING_RATE: ([\d.e-]+)", content).group(1)),
            'batch_size': int(re.search(r"BATCH_SIZE: (\d+)", content).group(1)),
            'max_length': int(re.search(r"MAX_LENGTH: (\d+)", content).group(1)),
            'patience': int(re.search(r"PATIENCE: (\d+)", content).group(1)),
        }
        test_f1_matches = re.findall(r"Test F1-Macro: ([\d.]+)", content)
        if not test_f1_matches: return None

        metrics = {'test_f1_macro': float(test_f1_matches[-1])}
        return {'config': config, 'metrics': metrics}
    except (AttributeError, IndexError):
        return None

def main():
    """Finds, parses, and prints a sorted summary of experiment results."""
    base_dir = Path("./experiments")
    all_results = []

    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' not found.")
        return

    for log_file in base_dir.glob("**/summary.log"):
        result = parse_summary_log(log_file)
        if result:
            all_results.append(result)

    if not all_results:
        print("No completed experiment logs found.")
        return

    sorted_results = sorted(
        all_results,
        key=lambda x: x['metrics']['test_f1_macro'],
        reverse=True
    )

    print("\n Grid Search Results (Sorted by Test F1-Macro)")
    for res in sorted_results:
        cfg = res['config']
        met = res['metrics']
        print(
            f"ID: {cfg['run_id']:<3} | "
            f"F1: {met['test_f1_macro']:.4f} | "
            f"LR: {cfg['lr']:.0e} | "
            f"BS: {cfg['batch_size']:<2} | "
            f"Len: {cfg['max_length']:<3} | "
            f"Pat: {cfg['patience']} | "
            f"Model: {cfg['model']}"
        )

if __name__ == '__main__':
    main()