import os
import re
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures
import argparse
import logging
from datetime import datetime
import gc
import json
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"preprocess_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def find_accident_timestamp(report_file):
    """
    Parse the TransientReport file to find the timestamp of an accident (Reactor Scram or Core Meltdown)
    Returns the timestamp in seconds or None if no accident found
    """
    try:
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Look for Reactor Scram or Core Meltdown events
        scram_match = re.search(r'(\d+\.?\d*)\s*sec,\s*Reactor\s*Scram', content)
        meltdown_match = re.search(r'(\d+\.?\d*)\s*sec,\s*Core\s*Meltdown', content)
        
        # Use the first match found (if any)
        if scram_match:
            timestamp = float(scram_match.group(1))
            return timestamp, "Reactor Scram"
        elif meltdown_match:
            timestamp = float(meltdown_match.group(1))
            return timestamp, "Core Meltdown"
        else:
            return None, None
    except Exception as e:
        logging.error(f"Error processing {report_file}: {e}")
        return None, None

def add_labels_to_csv(csv_file, accident_time, accident_type):
    """
    Add a 'label' column to the CSV file based on the accident timestamp
    0 = normal operation (before accident_time - 180 seconds)
    1 = potential accident (after that point)
    """
    try:
        # Read only the TIME column first to determine which rows to process
        df_time = pd.read_csv(csv_file, usecols=['TIME'])
        
        # Define threshold time (accident time minus 180 seconds)
        threshold_time = max(0, accident_time - 180)
        
        # Calculate labels based on TIME column
        labels = np.where(df_time['TIME'] < threshold_time, 0, 1)
        
        # Read the full CSV
        df = pd.read_csv(csv_file)
        
        # Add labels and accident information
        df['label'] = labels
        df['accident_timestamp'] = accident_time
        df['accident_type'] = accident_type
        
        # Save the modified CSV
        df.to_csv(csv_file, index=False)
        
        # Get statistics about labeling
        n_normal = sum(labels == 0)
        n_accident = sum(labels == 1)
        
        return True, {
            'file': os.path.basename(csv_file),
            'accident_time': accident_time,
            'accident_type': accident_type,
            'threshold_time': threshold_time,
            'total_rows': len(df),
            'normal_rows': n_normal,
            'accident_rows': n_accident,
            'normal_percentage': n_normal / len(df) * 100 if len(df) > 0 else 0,
            'accident_percentage': n_accident / len(df) * 100 if len(df) > 0 else 0
        }
    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}")
        return False, {'file': os.path.basename(csv_file), 'error': str(e)}

def process_single_simulation(args):
    """Process a single simulation (csv file and its report file)"""
    csv_file, report_file = args
    base_name = os.path.basename(csv_file)
    stats = {
        'file': base_name,
        'processed': False,
        'has_report': os.path.exists(report_file),
        'has_accident': False,
        'labeled': False,
        'error': None
    }
    
    if not stats['has_report']:
        stats['error'] = f"No TransientReport file found"
        return stats
    
    # Find accident timestamp
    accident_time, accident_type = find_accident_timestamp(report_file)
    stats['has_accident'] = accident_time is not None
    stats['accident_time'] = accident_time
    stats['accident_type'] = accident_type
    
    if accident_time is not None:
        # Add labels to the CSV
        success, label_stats = add_labels_to_csv(csv_file, accident_time, accident_type)
        stats['labeled'] = success
        if success:
            stats.update(label_stats)
    else:
        stats['error'] = f"No accident found in report file"
    
    stats['processed'] = True
    return stats

def get_common_columns(csv_files, sample_size=100):
    """
    Find common columns across CSV files using a sample of files
    to prevent memory issues
    """
    # Sample files if there are too many
    if len(csv_files) > sample_size:
        import random
        sampled_files = random.sample(csv_files, sample_size)
    else:
        sampled_files = csv_files
    
    common_columns = None
    all_columns = set()
    file_columns = {}
    
    for csv_file in tqdm(sampled_files, desc="Analyzing CSV structure"):
        try:
            # Read only the header to save memory
            df = pd.read_csv(csv_file, nrows=0)
            columns = set(df.columns)
            file_columns[os.path.basename(csv_file)] = list(columns)
            
            if common_columns is None:
                common_columns = columns
            else:
                common_columns &= columns
            all_columns |= columns
        except Exception as e:
            logging.error(f"Error reading columns from {csv_file}: {e}")
    
    return common_columns, all_columns, file_columns

def align_single_csv(args):
    """Align a single CSV file to the standard column format"""
    csv_file, final_columns = args
    try:
        df = pd.read_csv(csv_file)
        
        # Add missing columns with NaN values
        for col in final_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Select only the final columns and save
        df = df[final_columns]
        df.to_csv(csv_file, index=False)
        
        # Free memory
        del df
        gc.collect()
        
        return {'file': os.path.basename(csv_file), 'success': True}
    except Exception as e:
        return {'file': os.path.basename(csv_file), 'success': False, 'error': str(e)}

def align_csv_dimensions(csv_files, max_workers=8):
    """
    Align dimensions of all CSV files by finding the common columns
    and ensuring all files have the same columns
    """
    try:
        logging.info(f"Finding common columns across {len(csv_files)} CSV files")
        common_columns, all_columns, file_columns = get_common_columns(csv_files)
        
        if not common_columns:
            logging.error("No common columns found across files")
            return False
        
        # Make sure 'TIME', 'label', 'accident_timestamp', and 'accident_type' are included
        required_columns = {'TIME', 'label', 'accident_timestamp', 'accident_type'}
        final_columns = list(common_columns | required_columns)
        
        logging.info(f"Common columns: {len(common_columns)}/{len(all_columns)}")
        logging.info(f"Final column set: {len(final_columns)} columns")
        
        # Save column information for reference - Use the custom JSON encoder
        with open('column_info.json', 'w') as f:
            json.dump({
                'common_columns': list(common_columns),
                'all_columns': list(all_columns),
                'final_columns': final_columns,
                'file_columns': file_columns
            }, f, indent=2, cls=NumpyEncoder)
        
        # Process files in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [(csv_file, final_columns) for csv_file in csv_files]
            for result in tqdm(executor.map(align_single_csv, args), total=len(args), desc="Aligning CSV dimensions"):
                results.append(result)
        
        # Count successes and failures
        successes = sum(1 for r in results if r['success'])
        failures = len(results) - successes
        
        logging.info(f"Alignment complete: {successes} succeeded, {failures} failed")
        
        # Save alignment results - Use the custom JSON encoder
        with open('alignment_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        return True
    except Exception as e:
        logging.error(f"Error in align_csv_dimensions: {e}")
        return False

def process_all_simulations(root_dir, max_workers=8, align=True):
    """
    Process all simulations in the given directory structure
    """
    # Find all simulation folders
    simulation_dirs = []
    for dirpath, _, _ in os.walk(root_dir):
        if os.path.basename(dirpath) != "NPPAD" and dirpath != root_dir:  # Skip the root directory
            simulation_dirs.append(dirpath)
    
    if not simulation_dirs:
        simulation_dirs = [root_dir]  # Use root directory if no subdirectories found
    
    logging.info(f"Found {len(simulation_dirs)} simulation directories")
    
    # Collect all CSV and report file pairs
    all_pairs = []
    for sim_dir in simulation_dirs:
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(sim_dir, "*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files in {sim_dir}")
        
        for csv_file in csv_files:
            # Construct the corresponding TransientReport filename
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            report_file = os.path.join(sim_dir, f"{base_name}Transient Report.txt")
            all_pairs.append((csv_file, report_file))
    
    logging.info(f"Total simulations to process: {len(all_pairs)}")
    
    # Process simulations in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_single_simulation, all_pairs), total=len(all_pairs), desc="Processing simulations"):
            results.append(result)
    
    # Compute statistics
    total_processed = len(results)
    has_report = sum(1 for r in results if r['has_report'])
    has_accident = sum(1 for r in results if r['has_accident'])
    successfully_labeled = sum(1 for r in results if r['labeled'])
    
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"Files with reports: {has_report} ({has_report/total_processed*100:.2f}%)")
    logging.info(f"Files with accidents: {has_accident} ({has_accident/total_processed*100:.2f}%)")
    logging.info(f"Successfully labeled: {successfully_labeled} ({successfully_labeled/total_processed*100:.2f}%)")
    
    # Save detailed results - Use the custom JSON encoder
    with open('processing_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Do CSV alignment if requested
    if align and successfully_labeled > 0:
        all_csv_files = [pair[0] for pair in all_pairs]
        logging.info(f"Aligning dimensions for {len(all_csv_files)} CSV files")
        align_csv_dimensions(all_csv_files, max_workers=max_workers)
    
    return {
        'total_processed': total_processed,
        'has_report': has_report,
        'has_accident': has_accident,
        'successfully_labeled': successfully_labeled
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess nuclear power plant accident simulations")
    parser.add_argument("--root-dir", default="NPPAD", help="Root directory containing simulations")
    parser.add_argument("--workers", type=int, default=8, help="Maximum number of worker processes")
    parser.add_argument("--no-align", action="store_true", help="Skip CSV alignment step")
    parser.add_argument("--align-only", action="store_true", help="Only perform CSV alignment")
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root_dir)
    
    if not os.path.exists(root_dir):
        logging.error(f"Error: Directory {root_dir} not found")
    else:
        logging.info(f"Starting preprocessing of simulations in {root_dir}")
        
        if args.align_only:
            # Only perform CSV alignment
            all_csv_files = []
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.endswith('.csv'):
                        all_csv_files.append(os.path.join(dirpath, filename))
            
            logging.info(f"Aligning dimensions for {len(all_csv_files)} CSV files")
            align_csv_dimensions(all_csv_files, max_workers=args.workers)
        else:
            # Process simulations
            stats = process_all_simulations(root_dir, max_workers=args.workers, align=not args.no_align)
            logging.info(f"Processing summary: {stats}")
        
        logging.info("Preprocessing complete!") 