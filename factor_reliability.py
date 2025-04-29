#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def cronbachs_alpha(item_scores):
    """
    Calculate Cronbach's alpha for a set of item scores on a single factor.
    Vectorized implementation for improved performance.
    
    Parameters:
    -----------
    item_scores : numpy.ndarray
        Matrix of scores, where rows are models and columns are questions/items
        
    Returns:
    --------
    float : Cronbach's alpha coefficient
    """
    # Ensure item_scores is 2D
    if len(item_scores.shape) == 1:
        item_scores = item_scores.reshape(-1, 1)
    
    # Check if we have enough items
    n_items = item_scores.shape[1]
    if n_items <= 1:
        print("Warning: Need at least 2 items to calculate Cronbach's alpha")
        return np.nan
    
    # Fast vectorized implementation
    try:
        # Handle NaN values with a masked array
        scores = np.ma.masked_array(item_scores, np.isnan(item_scores))
        
        # Count valid (non-NaN) items per row
        valid_counts = scores.count(axis=1)
        
        # Only keep rows with enough data
        valid_rows = valid_counts >= 2
        if np.sum(valid_rows) < 2:
            print("Warning: Not enough valid data to calculate Cronbach's alpha")
            return np.nan
            
        # Calculate item variances for valid entries
        item_vars = np.ma.var(scores, axis=0, ddof=1)
        
        # Calculate total score and its variance
        total_scores = np.ma.sum(scores, axis=1)
        total_var = np.ma.var(total_scores, ddof=1)
        
        # Cronbach's alpha formula
        if total_var == 0:
            return np.nan
            
        # Classical formula: alpha = (k / (k - 1)) * (1 - (sum(item_vars) / total_var))
        n_items = scores.shape[1]
        return (n_items / (n_items - 1)) * (1 - np.sum(item_vars) / total_var)
        
    except Exception as e:
        # Try alternative implementation with pairwise correlation if first method fails
        try:
            # Remove rows with all NaNs
            mask = ~np.isnan(item_scores).all(axis=1)
            scores = item_scores[mask]
            
            # Calculate all pairwise correlations
            valid_pairs = 0
            sum_corr = 0
            
            # Vectorized correlation calculation
            for i in range(n_items):
                for j in range(i+1, n_items):
                    mask_ij = ~(np.isnan(scores[:, i]) | np.isnan(scores[:, j]))
                    if np.sum(mask_ij) >= 3:  # Need at least 3 valid pairs
                        x = scores[mask_ij, i]
                        y = scores[mask_ij, j]
                        corr = np.corrcoef(x, y)[0, 1]
                        if not np.isnan(corr):
                            sum_corr += corr
                            valid_pairs += 1
            
            if valid_pairs == 0:
                return np.nan
                
            # Average correlation
            avg_corr = sum_corr / valid_pairs
            
            # Spearman-Brown formula: alpha = (n*avg_r)/(1 + (n-1)*avg_r)
            return (n_items * avg_corr) / (1 + (n_items - 1) * avg_corr)
            
        except Exception as nested_e:
            print(f"Error calculating Cronbach's alpha: {e} -> {nested_e}")
            return np.nan

def cross_loadings(factor_scores):
    """
    Analyze cross-loadings to assess discriminant validity.
    
    Parameters:
    -----------
    factor_scores : dict
        Dictionary where keys are factor names and values are matrices of scores
        
    Returns:
    --------
    pandas.DataFrame : Matrix of cross-loadings
    float : Mean ratio of primary-to-secondary loadings (higher is better)
    """
    # Extract factor names and prepare data
    factor_names = list(factor_scores.keys())
    n_factors = len(factor_names)
    
    if n_factors <= 1:
        print("Warning: Need at least 2 factors to calculate cross-loadings")
        return pd.DataFrame(), np.nan
    
    # Create dataframe to store loadings
    loadings_df = pd.DataFrame(index=factor_names, columns=factor_names)
    
    # Get average score for each factor (across items), handling NaNs
    factor_avgs = {}
    for factor in factor_names:
        if len(factor_scores[factor].shape) > 1:
            # Calculate mean along rows, ignoring NaNs
            factor_avgs[factor] = np.nanmean(factor_scores[factor], axis=1)
        else:
            factor_avgs[factor] = factor_scores[factor]
    
    # Calculate loadings (correlations)
    for factor1 in factor_names:
        for factor2 in factor_names:
            try:
                # Handle NaNs when calculating correlation
                # Get indices where both factors have non-NaN values
                mask = ~np.isnan(factor_avgs[factor1]) & ~np.isnan(factor_avgs[factor2])
                if np.sum(mask) < 3:  # Need at least 3 points for meaningful correlation
                    loadings_df.loc[factor1, factor2] = np.nan
                    continue
                
                corr, _ = pearsonr(factor_avgs[factor1][mask], factor_avgs[factor2][mask])
                loadings_df.loc[factor1, factor2] = corr
            except Exception as e:
                print(f"Error calculating correlation between {factor1} and {factor2}: {e}")
                loadings_df.loc[factor1, factor2] = np.nan
    
    # Calculate ratio of primary to secondary loadings
    primary_loadings = np.diag(loadings_df)
    
    # For each factor, find the highest cross-loading
    secondary_loadings = []
    for i, factor in enumerate(factor_names):
        # Get all loadings except the diagonal element
        cross_loads = loadings_df.loc[factor].drop(factor).abs()
        if not cross_loads.empty and not cross_loads.isna().all():
            secondary_loadings.append(cross_loads.max())
        else:
            secondary_loadings.append(np.nan)
    
    # Compute ratio (primary / secondary)
    ratios = []
    for i in range(len(primary_loadings)):
        if i < len(secondary_loadings) and not np.isnan(secondary_loadings[i]) and secondary_loadings[i] > 0:
            if not np.isnan(primary_loadings[i]):
                ratios.append(abs(primary_loadings[i]) / secondary_loadings[i])
    
    # Mean ratio (higher is better for discriminant validity)
    if ratios:
        mean_ratio = np.mean(ratios)
    else:
        mean_ratio = np.nan
    
    return loadings_df, mean_ratio

def htmt_ratio_f(factor1_scores, factor2_scores):
    """
    Calculate Heterotrait-Monotrait ratio between two factors.
    Fast implementation using vectorized operations.
    
    Parameters:
    -----------
    factor1_scores : numpy.ndarray
        Matrix of scores for factor 1
    factor2_scores : numpy.ndarray
        Matrix of scores for factor 2
        
    Returns:
    --------
    float : HTMT ratio (lower is better, < 0.85 indicates good discriminant validity)
    """
    # Ensure we have 2D arrays
    if len(factor1_scores.shape) == 1:
        factor1_scores = factor1_scores.reshape(-1, 1)
    if len(factor2_scores.shape) == 1:
        factor2_scores = factor2_scores.reshape(-1, 1)
    
    # Check if we have enough items
    if factor1_scores.shape[1] <= 1 or factor2_scores.shape[1] <= 1:
        print("Warning: Need at least 2 items per factor to calculate HTMT ratio")
        return np.nan
    
    try:
        # Get dimensions
        n_models = factor1_scores.shape[0]
        n_items1 = factor1_scores.shape[1]
        n_items2 = factor2_scores.shape[1]
        
        # Calculate correlation matrices for within-factor correlations
        # First create arrays to store all pairwise correlations
        within1_corrs = []
        within2_corrs = []
        between_corrs = []
        
        # Within factor 1 correlations - using matrix operations for efficiency
        for i in range(n_items1):
            for j in range(i+1, n_items1):
                # Get valid observation mask
                mask = ~np.isnan(factor1_scores[:, i]) & ~np.isnan(factor1_scores[:, j])
                if np.sum(mask) < 3:  # Need at least 3 valid observations
                    continue
                
                # Extract valid observations and calculate correlation
                x1 = factor1_scores[mask, i]
                x2 = factor1_scores[mask, j]
                
                # Calculate correlation directly
                corr = np.corrcoef(x1, x2)[0, 1]
                if not np.isnan(corr):
                    within1_corrs.append(corr)
        
        # Within factor 2 correlations
        for i in range(n_items2):
            for j in range(i+1, n_items2):
                # Get valid observation mask
                mask = ~np.isnan(factor2_scores[:, i]) & ~np.isnan(factor2_scores[:, j])
                if np.sum(mask) < 3:
                    continue
                
                # Extract valid observations and calculate correlation
                x1 = factor2_scores[mask, i]
                x2 = factor2_scores[mask, j]
                
                # Calculate correlation directly
                corr = np.corrcoef(x1, x2)[0, 1]
                if not np.isnan(corr):
                    within2_corrs.append(corr)
        
        # Between-factor correlations - all pairs of items across the two factors
        # This is the heterotrait part
        for i in range(n_items1):
            for j in range(n_items2):
                # Get valid observation mask
                mask = ~np.isnan(factor1_scores[:, i]) & ~np.isnan(factor2_scores[:, j])
                if np.sum(mask) < 3:
                    continue
                
                # Extract valid observations and calculate correlation
                x1 = factor1_scores[mask, i]
                x2 = factor2_scores[mask, j]
                
                # Calculate correlation directly
                corr = np.corrcoef(x1, x2)[0, 1]
                if not np.isnan(corr):
                    between_corrs.append(corr)
        
        # Check if we have enough valid correlations
        if not within1_corrs or not within2_corrs or not between_corrs:
            return np.nan
        
        # Calculate means of correlations
        within1_mean = np.mean(within1_corrs)
        within2_mean = np.mean(within2_corrs)
        between_mean = np.mean(between_corrs)
        
        # Avoid division by zero or negative values
        if within1_mean <= 0 or within2_mean <= 0:
            return np.nan
        
        # Calculate HTMT ratio
        htmt = between_mean / np.sqrt(within1_mean * within2_mean)
        return htmt
        
    except Exception as e:
        print(f"Error calculating HTMT ratio: {e}")
        return np.nan

def calculate_factor_reliability(factor_scores):
    """
    Calculate reliability and discriminant validity metrics for each factor.
    
    Parameters:
    -----------
    factor_scores : dict
        Dictionary where keys are factor names and values are matrices of scores
        
    Returns:
    --------
    pandas.DataFrame : Reliability metrics for each factor
    """
    from tqdm.auto import tqdm
    
    # Extract factor names
    factor_names = list(factor_scores.keys())
    n_factors = len(factor_names)
    
    # Create dataframe to store results
    results = pd.DataFrame(index=factor_names, 
                          columns=["cronbachs_alpha", "cross_loading_ratio", 
                                  "avg_htmt_ratio", "reliability_score"])
    
    # Calculate Cronbach's alpha for each factor
    print("Calculating Cronbach's alpha for each factor...")
    for factor in tqdm(factor_names, desc="Cronbach's Alpha"):
        alpha = cronbachs_alpha(factor_scores[factor])
        results.loc[factor, "cronbachs_alpha"] = alpha
    
    # Calculate cross-loadings
    if n_factors > 1:
        loadings_df, mean_ratio = cross_loadings(factor_scores)
        for factor in factor_names:
            results.loc[factor, "cross_loading_ratio"] = mean_ratio
    else:
        for factor in factor_names:
            results.loc[factor, "cross_loading_ratio"] = np.nan
    
    # Calculate HTMT ratios
    if n_factors > 1:
        print("Calculating HTMT ratios between factors...")
        
        # Calculate all pairwise HTMT ratios
        all_htmt_ratios = {}
        total_pairs = n_factors * (n_factors - 1) // 2
        with tqdm(total=total_pairs, desc="HTMT Ratios") as progress_bar:
            for i, factor1 in enumerate(factor_names):
                for factor2 in factor_names[i+1:]:
                    # Calculate HTMT ratio using the optimized function
                    ratio = htmt_ratio_f(factor_scores[factor1], factor_scores[factor2])
                    
                    # Store ratio
                    if not np.isnan(ratio):
                        key = (factor1, factor2)
                        all_htmt_ratios[key] = ratio
                    
                    progress_bar.update(1)
        
        # For each factor, compute average HTMT ratio with all other factors
        for factor in factor_names:
            htmt_ratios = []
            for other_factor in factor_names:
                if factor == other_factor:
                    continue
                    
                # Get the correct key order
                if factor < other_factor:
                    key = (factor, other_factor)
                else:
                    key = (other_factor, factor)
                    
                # Add to list if we have this ratio
                if key in all_htmt_ratios:
                    htmt_ratios.append(all_htmt_ratios[key])
            
            # Lower HTMT is better, so we'll use 1 - avg_htmt for the combined score
            if htmt_ratios:
                avg_htmt = np.mean(htmt_ratios)
                results.loc[factor, "avg_htmt_ratio"] = avg_htmt
            else:
                results.loc[factor, "avg_htmt_ratio"] = np.nan
    else:
        for factor in factor_names:
            results.loc[factor, "avg_htmt_ratio"] = np.nan
    
    # Calculate combined reliability score
    # Higher alpha is better, higher cross-loading ratio is better, lower HTMT is better
    for factor in factor_names:
        alpha = results.loc[factor, "cronbachs_alpha"]
        cross_ratio = results.loc[factor, "cross_loading_ratio"]
        htmt_ratio = results.loc[factor, "avg_htmt_ratio"]
        
        # Combine metrics, handling NaNs
        valid_metrics = []
        
        if not np.isnan(alpha):
            # Scale to 0-1 range (most alphas are between 0.5-0.95)
            scaled_alpha = min(max((alpha - 0.5) / 0.45, 0), 1)
            valid_metrics.append(scaled_alpha)
        
        if not np.isnan(cross_ratio):
            # Cross-loading ratio is already in good range
            valid_metrics.append(min(cross_ratio / 2, 1))  # Cap at 1.0
        
        if not np.isnan(htmt_ratio):
            # Convert HTMT so lower is better (0.85 is typical threshold)
            htmt_score = min(max(1 - htmt_ratio, 0), 1)
            valid_metrics.append(htmt_score)
        
        # Calculate combined score
        if valid_metrics:
            reliability_score = np.mean(valid_metrics)
            results.loc[factor, "reliability_score"] = reliability_score
        else:
            results.loc[factor, "reliability_score"] = np.nan
    
    return results

def bootstrap_factor_reliability(factor_scores, n_bootstrap=1000):
    """
    Use bootstrap to estimate confidence intervals for factor reliability metrics.
    
    Parameters:
    -----------
    factor_scores : dict
        Dictionary where keys are factor names and values are matrices of scores
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns:
    --------
    dict : Dictionary with reliability metrics and confidence intervals
    """
    from tqdm.auto import tqdm
    # Extract factor names
    factor_names = list(factor_scores.keys())
    n_models = next(iter(factor_scores.values())).shape[0]
    
    # Store bootstrap results
    bootstrap_results = {
        factor: {
            "reliability_scores": [],
            "cronbachs_alpha": [],
            "cross_loading_ratio": [],
            "avg_htmt_ratio": []
        } for factor in factor_names
    }
    
    # Perform bootstrap
    print(f"Performing {n_bootstrap} bootstrap iterations to compute confidence intervals...")
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap Progress"):
        # Sample models with replacement
        indices = np.random.choice(n_models, size=n_models, replace=True)
        
        # Create bootstrap sample
        bootstrap_scores = {
            factor: scores[indices] for factor, scores in factor_scores.items()
        }
        
        # Calculate reliability metrics for this sample (with silent mode)
        # Temporarily redirect stdout to suppress progress bars inside the bootstrap
        import sys
        from io import StringIO
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            sample_results = calculate_factor_reliability(bootstrap_scores)
        finally:
            sys.stdout = original_stdout
        
        # Store results
        for factor in factor_names:
            bootstrap_results[factor]["reliability_scores"].append(
                sample_results.loc[factor, "reliability_score"])
            bootstrap_results[factor]["cronbachs_alpha"].append(
                sample_results.loc[factor, "cronbachs_alpha"])
            bootstrap_results[factor]["cross_loading_ratio"].append(
                sample_results.loc[factor, "cross_loading_ratio"])
            bootstrap_results[factor]["avg_htmt_ratio"].append(
                sample_results.loc[factor, "avg_htmt_ratio"])
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for factor in factor_names:
        confidence_intervals[factor] = {}
        
        for metric in ["reliability_scores", "cronbachs_alpha", "cross_loading_ratio", "avg_htmt_ratio"]:
            values = bootstrap_results[factor][metric]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                confidence_intervals[factor][metric] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "lower": np.percentile(values, 2.5),
                    "upper": np.percentile(values, 97.5)
                }
            else:
                confidence_intervals[factor][metric] = {
                    "mean": np.nan,
                    "median": np.nan,
                    "lower": np.nan,
                    "upper": np.nan
                }
    
    return confidence_intervals

def load_factor_scores_from_jsonl(directory_path):
    """
    Load factor scores from JSONL files in the specified directory.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing JSONL files
        
    Returns:
    --------
    dict : Dictionary mapping factor names to score matrices
    """
    import json
    import glob
    
    # Find all JSONL files (excluding those ending with _ct.txt)
    jsonl_files = [f for f in glob.glob(os.path.join(directory_path, "*.jsonl")) 
                  if not f.endswith("_ct.txt")]
    
    if not jsonl_files:
        print(f"No JSONL files found in {directory_path}")
        return None
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # First scan all files to discover all models, questions and factors
    # This is to ensure we initialize our data structures properly
    all_model_names = set()
    all_question_ids = set()
    all_factor_keys = set()
    question_counts = {}
    
    for jsonl_file in jsonl_files:
        model_name = os.path.basename(jsonl_file).replace(".jsonl", "")
        all_model_names.add(model_name)
        
        # Read the file
        try:
            with open(jsonl_file, 'r') as f:
                model_data = [json.loads(line) for line in f]
            
            # Extract question IDs and count
            file_question_ids = []
            for item in model_data:
                if "question_id" in item:
                    qid = item["question_id"]
                    all_question_ids.add(qid)
                    file_question_ids.append(qid)
                
                # Check if "games" field exists and discover factors
                if "games" in item:
                    for game in item["games"]:
                        for key in game:
                            if "_score" in key:
                                all_factor_keys.add(key)
            
            question_counts[model_name] = len(file_question_ids)
        except Exception as e:
            print(f"Error reading {jsonl_file}: {e}")
    
    print(f"Found {len(all_model_names)} models, {len(all_question_ids)} questions, and {len(all_factor_keys)} factor keys")
    
    # Create a sorted list of all question IDs (to ensure consistent ordering)
    all_question_ids = sorted(list(all_question_ids))
    all_model_names = sorted(list(all_model_names))
    
    # Create a dictionary mapping question IDs to indices
    question_id_to_idx = {qid: i for i, qid in enumerate(all_question_ids)}
    
    # Initialize the factor data structure
    factor_data = {}
    for factor_key in all_factor_keys:
        factor_data[factor_key] = {}
        for model_name in all_model_names:
            # Initialize with NaN values for all questions
            factor_data[factor_key][model_name] = [np.nan] * len(all_question_ids)
    
    # Now process each file and fill in the values
    for jsonl_file in jsonl_files:
        model_name = os.path.basename(jsonl_file).replace(".jsonl", "")
        
        try:
            # Read the file
            with open(jsonl_file, 'r') as f:
                model_data = [json.loads(line) for line in f]
            
            # Process each question
            for item in model_data:
                if "question_id" not in item or "games" not in item:
                    continue
                
                question_id = item["question_id"]
                question_idx = question_id_to_idx.get(question_id)
                
                if question_idx is None:
                    continue  # Skip if question ID not in our master list
                
                # Process each game (judgment)
                for game in item["games"]:
                    # Extract scores from the game
                    for key, value in game.items():
                        if key in all_factor_keys:
                            # Convert score to numeric value
                            if isinstance(value, str):
                                # Mapping for letter scores like A>B, A=B, etc.
                                score_mapping = {
                                    '': 3,
                                    'A>>B': 1,
                                    'A>B': 2,
                                    'A=B': 3,
                                    'B>A': 4,
                                    'B>>A': 5,
                                    'A<<B': 5,
                                    'A<B': 4,
                                    'B=A': 3,
                                    'B<A': 2,
                                    'B<<A': 1
                                }
                                numeric_value = score_mapping.get(value, 3)
                            else:
                                numeric_value = value
                            
                            # Store the value
                            factor_data[key][model_name][question_idx] = numeric_value
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue
    
    # Convert to numpy arrays
    result = {}
    for factor, model_scores in factor_data.items():
        # Create a matrix with rows=models, columns=questions
        matrix = np.array([model_scores[model] for model in all_model_names])
        
        # Store in result
        result[factor] = matrix
        
        # Print some diagnostics
        nan_count = np.isnan(matrix).sum()
        total_elements = matrix.size
        print(f"Factor {factor}: {nan_count}/{total_elements} NaN values ({100 * nan_count / total_elements:.1f}%)")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Calculate factor reliability metrics from model judgment data.')
    parser.add_argument('directory', type=str, help='Directory containing processed JSONL files with judgment scores')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files (defaults to the input directory)')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap samples for confidence intervals')
    parser.add_argument('--skip-bootstrap', action='store_true', help='Skip bootstrap confidence interval calculation')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer bootstrap samples)')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load factor scores
    print(f"Loading factor scores from {args.directory}...")
    factor_scores = load_factor_scores_from_jsonl(args.directory)
    
    if factor_scores is None:
        print("Failed to load factor scores. Exiting.")
        return
    
    print(f"Loaded scores for {len(factor_scores)} factors:")
    for factor, scores in factor_scores.items():
        print(f"  {factor}: {scores.shape[0]} models, {scores.shape[1]} questions")
    
    # Calculate factor reliability
    print("\nCalculating factor reliability metrics...")
    reliability_metrics = calculate_factor_reliability(factor_scores)
    
    # Save reliability metrics
    metrics_path = os.path.join(output_dir, "factor_reliability_metrics.csv")
    reliability_metrics.to_csv(metrics_path)
    print(f"Saved reliability metrics to {metrics_path}")
    
    # Calculate bootstrap confidence intervals
    confidence_intervals = None
    if args.skip_bootstrap:
        print("\nSkipping bootstrap confidence intervals as requested.")
    else:
        # Adjust bootstrap samples for quick mode
        n_bootstrap = 100 if args.quick else args.n_bootstrap
        print(f"\nCalculating bootstrap confidence intervals with {n_bootstrap} samples...")
        confidence_intervals = bootstrap_factor_reliability(factor_scores, n_bootstrap)
        
        # Prepare confidence intervals for saving
        ci_data = []
        for factor, metrics in confidence_intervals.items():
            for metric, values in metrics.items():
                ci_data.append({
                    "factor": factor,
                    "metric": metric,
                    "mean": values["mean"],
                    "median": values["median"],
                    "lower_ci": values["lower"],
                    "upper_ci": values["upper"]
                })
    
    # Save confidence intervals if they were calculated
    if not args.skip_bootstrap:
        ci_df = pd.DataFrame(ci_data)
        ci_path = os.path.join(output_dir, "factor_reliability_confidence_intervals.csv")
        ci_df.to_csv(ci_path, index=False)
        print(f"Saved confidence intervals to {ci_path}")
    
    # Print summary of reliability scores
    print("\nReliability Score Summary (higher is better):")
    for factor in reliability_metrics.index:
        reliability = reliability_metrics.loc[factor, "reliability_score"]
        if not np.isnan(reliability):
            if confidence_intervals and factor in confidence_intervals:
                ci = confidence_intervals[factor]["reliability_scores"]
                print(f"  {factor}: {reliability:.3f} (95% CI: {ci['lower']:.3f}-{ci['upper']:.3f})")
            else:
                print(f"  {factor}: {reliability:.3f}")
    
    # Suggest interpretation
    print("\nInterpretation Guide:")
    print("  Reliability Score > 0.8: Excellent - factor is consistent and distinct")
    print("  Reliability Score 0.6-0.8: Good - factor is generally reliable")
    print("  Reliability Score 0.4-0.6: Moderate - factor may need refinement")
    print("  Reliability Score < 0.4: Poor - factor may not be reliably measured")

if __name__ == "__main__":
    main()