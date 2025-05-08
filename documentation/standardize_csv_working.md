# CSV Standardization Script Documentation

## Overview
This script standardizes the structure of CSV files by keeping only the columns that are common across all files. It maintains the TIME column as the first column and arranges all other columns lexicographically.

## Features
- Recursively finds all CSV files in the current directory and subdirectories
- Identifies columns that are common across all CSV files
- Drops any columns that are not present in all files
- Standardizes column ordering while preserving data
- Maintains TIME as the first column
- Generates detailed logs of the standardization process

## Script Structure

### Main Functions

1. `get_all_csv_files(base_dir)`
   - Recursively finds all CSV files in the specified directory
   - Returns a list of file paths

2. `get_common_columns(csv_files)`
   - Reads all CSV files to identify columns that are present in every file
   - Returns a sorted list of common column names
   - Uses set intersection to find columns present in all files

3. `standardize_csv(file_path, common_columns)`
   - Processes a single CSV file
   - Keeps only the common columns
   - Ensures TIME column is first
   - Saves the standardized file

### Logging
- Creates a log file in the `logs` directory
- Logs information about:
  - Number of CSV files found
  - Number of common columns identified
  - List of common columns found
  - Success/failure of processing each file
  - Any errors encountered during processing

## Usage
1. Place the script in the root directory containing your CSV files
2. Run the script:
   ```bash
   python standardize_csv.py
   ```

## Output
- All CSV files will be updated in place with only common columns
- Column ordering will be standardized with TIME first
- A log file will be created at `logs/standardize_csv.log`

## Error Handling
- The script handles errors gracefully and logs them
- If a file cannot be read or processed, the error is logged and the script continues with the next file
- If the first file cannot be read, the script will exit as it cannot determine common columns 