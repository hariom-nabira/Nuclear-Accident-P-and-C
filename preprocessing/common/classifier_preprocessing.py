import os
import pandas as pd
import shutil
import logging
import re
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/classifier_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def find_accident_timestamp(transient_report_path):
    """
    Find the timestamp of Reactor Scram or Core Meltdown from transient report.
    Returns None if no accident found.
    """
    try:
        with open(transient_report_path, 'r') as f:
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
        logging.error(f"Error reading transient report {transient_report_path}: {str(e)}")
        return None, None

def preprocess_simulation(csv_path, transient_report_path, output_dir):
    """
    Preprocess a single simulation by truncating data after accident timestamp.
    If no accident is found, keep the original data as is.
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Find accident timestamp
        accident_time, accident_type = find_accident_timestamp(transient_report_path)
        if accident_time is None:
            logging.info(f"No accident found in {transient_report_path}, keeping original data...")
            # Save original data
            output_path = os.path.join(output_dir, os.path.basename(csv_path))
            df.to_csv(output_path, index=False)
            logging.info(f"Successfully copied {csv_path} (No accident found)")
            return True
        
        # Truncate data
        df = df[df['TIME'] <= accident_time]
        
        # Save preprocessed data
        output_path = os.path.join(output_dir, os.path.basename(csv_path))
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully preprocessed {csv_path} (Accident: {accident_type} at {accident_time} seconds)")
        return True
        
    except Exception as e:
        logging.error(f"Error preprocessing {csv_path}: {str(e)}")
        return False

def get_simulation_files(accident_dir):
    """
    Get all simulation files in an accident directory.
    Returns a list of (severity, csv_file, transient_file) tuples.
    """
    simulations = []
    for file in os.listdir(accident_dir):
        if file.endswith('.csv'):
            severity = file.split('.')[0]
            transient_file = f"{severity}Transient Report.txt"
            if os.path.exists(os.path.join(accident_dir, transient_file)):
                simulations.append((severity, file, transient_file))
    return sorted(simulations, key=lambda x: int(x[0]))  # Sort by severity number

def main():
    """
    Main function to preprocess all simulations.
    """
    # Source and destination directories
    source_dir = "NPPAD"
    dest_dir = "NPPAD_for_classifiers"
    
    # Create destination directory if it doesn't exist
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    # Process each accident type directory
    for accident_type in os.listdir(source_dir):
        accident_dir = os.path.join(source_dir, accident_type)
        if not os.path.isdir(accident_dir):
            continue
            
        logging.info(f"Processing accident type: {accident_type}")
        
        # Create corresponding directory in destination
        dest_accident_dir = os.path.join(dest_dir, accident_type)
        os.makedirs(dest_accident_dir)
        
        # Get all simulation files
        simulations = get_simulation_files(accident_dir)
        logging.info(f"Found {len(simulations)} simulations for {accident_type}")
        
        # Process each simulation
        for severity, csv_file, transient_file in simulations:
            csv_path = os.path.join(accident_dir, csv_file)
            transient_path = os.path.join(accident_dir, transient_file)
            
            preprocess_simulation(csv_path, transient_path, dest_accident_dir)
    
    logging.info("Preprocessing completed!")

if __name__ == "__main__":
    main() 