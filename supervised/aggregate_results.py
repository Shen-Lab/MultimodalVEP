#!/usr/bin/env python3

"""

Aggregate evaluation results and calculate average Spearman correlations

Deduplicates by protein name, keeping the last (most recent) result per protein.

"""

import re

import numpy as np



def parse_spearman_from_file(filepath):

    """Extract Spearman correlations from a result file, deduplicating by protein"""

    protein_spearmans = {}  # protein_name -> spearman value (last one wins)

    current_protein = None

    

    with open(filepath, 'r') as f:

        for line in f:

            # Match DMS ID line

            dms_match = re.search(r'DMS ID:\s+(.+)', line)

            if dms_match:

                current_protein = dms_match.group(1).strip().split("/")[-1].replace(".csv", "")

            

            # Match Spearman value (skips nan automatically)

            spearman_match = re.search(r'Spearman Correlation:\s+([-+]?\d*\.\d+|\d+)', line)

            if spearman_match and current_protein:

                protein_spearmans[current_protein] = float(spearman_match.group(1))

    

    return protein_spearmans



def aggregate_results(model_name, results_dir):

    """Aggregate results across all folds for a given model"""

    print(f"\n{'='*60}")

    print(f"Aggregating {model_name} Results")

    print(f"{'='*60}\n")

    

    fold_averages = []

    all_protein_spearmans = []

    

    for fold in range(5):

        result_file = f"{results_dir}/fold{fold}_results.txt"

        protein_spearmans = parse_spearman_from_file(result_file)

        

        if protein_spearmans:

            values = list(protein_spearmans.values())

            fold_avg = np.mean(values)

            fold_averages.append(fold_avg)

            all_protein_spearmans.extend(values)

            print(f"Fold {fold}: {len(values)} proteins, Average Spearman = {fold_avg:.4f}")

        else:

            print(f"Fold {fold}: No results found!")

    

    if fold_averages:

        overall_mean = np.mean(fold_averages)

        overall_std = np.std(all_protein_spearmans, ddof=1)

        

        print(f"\n{'-'*60}")

        print(f"Overall Results for {model_name}:")

        print(f"  Total proteins evaluated: {len(all_protein_spearmans)}")

        print(f"  Mean Spearman (avg of fold avgs): {overall_mean:.4f}")

        print(f"  Std Deviation (across all proteins): {overall_std:.4f}")

        print(f"  Final Result: {overall_mean:.3f} ± {overall_std:.3f}")

        print(f"{'-'*60}\n")

        

        return overall_mean, overall_std

    else:

        print("ERROR: No results found!")

        return None, None



if __name__ == "__main__":

    esm2_mean, esm2_std = aggregate_results("ESM2", "../eval_results/esm2")

    saprot_mean, saprot_std = aggregate_results("SaProt", "../eval_results/saprot")

    

    print(f"\n{'='*60}")

    print("COMPARISON TO TARGET METRICS")

    print(f"{'='*60}\n")

    

    if esm2_mean is not None:

        print(f"ESM2 (supervised MLP on ESM2 embeddings):")

        print(f"  Your result: {esm2_mean:.3f} ± {esm2_std:.3f}")

        print(f"  Note: ProteinGym target 0.414 is ZERO-SHOT ESM2,")

        print(f"        not comparable to your supervised result.")

        print()

    

    if saprot_mean is not None:

        print(f"SaProt (supervised):")

        print(f"  Your result: {saprot_mean:.3f} ± {saprot_std:.3f}")

        print(f"  Target:      0.724 ± 0.190")

        print(f"  Difference:  {saprot_mean - 0.724:+.3f}")

        print()