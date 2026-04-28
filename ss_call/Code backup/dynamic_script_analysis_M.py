"""
============================================================
Dynamic Script Analysis Tool (Mandarin Version)
============================================================

This standalone tool analyzes script files to generate dynamic term importance weights.
It exports weights to CSV.gz format for external loading by dictionaries.py.
This version is specifically designed for Mandarin Chinese text processing.

Usage:
    python dynamic_script_analysis_M.py [excel_file] [sheet_name]

Features:
- Analyzes uniqueness of terms across all required discussion points
- Generates dynamic term importance weights based on rarity and theme relevance
- Only keeps top 16 words with highest uniqueness score per script
- Exports weights to CSV.gz format for lazy loading
- Uses jieba for Mandarin Chinese text segmentation
- Maintains point-specific weight structure for precise analysis

============================================================
"""

import pandas as pd
import numpy as np
import re
import jieba
import os
import sys
import argparse
from collections import Counter

# Import dictionaries
import dictionaries

def preprocess_text(text):
    """Enhanced text preprocessing for Mandarin and speech-to-text errors."""
    # Keep Chinese characters (simplified/traditional), English letters, and specified punctuation
    # Remove all other characters including spaces, newlines, tabs, etc.
    text = re.sub(r'[^\u4e00-\u9fffa-zA-Z,.。，。%()（）]+', '', str(text))
    # Normalize common speech-to-text errors
    for correct, variations in dictionaries.error_patterns.items():
        for variation in variations:
            text = text.replace(variation, correct)
    return text.strip()

def tokenize_text(text):
    """Tokenize text using jieba and filter out stopwords."""
    # 使用精确模式分词
    tokens = jieba.lcut(text)
    # Filter out stopwords and keep only business-relevant terms
    business_tokens = [token for token in tokens if token not in dictionaries.stopwords and len(token) > 1]
    return business_tokens

def deduplicate_scripts(required_points_df, excel_file, sheet_name):
    """
    Remove duplicate scripts and merge discussion points with identical scripts.
    
    Logic:
    - Group by Standard_Script content
    - For scripts that appear in multiple discussion points, merge those points
    - Combine discussion point names with ' + ' separator
    - Save merged results to new file if any duplicates found
    
    Args:
        required_points_df: DataFrame with Required_Discussion_Point and Standard_Script columns
        excel_file: Original Excel file name
        sheet_name: Original sheet name
        
    Returns:
        tuple: (deduplicated_df, duplicates_found, output_file_path)
    """
    print("=== STEP 0: SCRIPT DEDUPLICATION CHECK ===")
    
    # Check for duplicate scripts
    script_groups = required_points_df.groupby('Standard_Script')['Required_Discussion_Point'].apply(list).to_dict()
    
    duplicates_found = False
    merged_data = []
    
    for script, points in script_groups.items():
        if pd.isna(script):
            # Skip NaN scripts
            continue
            
        if len(points) > 1:
            duplicates_found = True
            # Merge discussion points
            merged_point = ' + '.join(sorted(points))
            merged_data.append({
                'Required_Discussion_Point': merged_point,
                'Standard_Script': script
            })
            print(f"📋 Merged duplicate script: {points} -> '{merged_point}'")
        else:
            # Keep single point as is
            merged_data.append({
                'Required_Discussion_Point': points[0],
                'Standard_Script': script
            })
    
    deduplicated_df = pd.DataFrame(merged_data)
    
    if duplicates_found:
        # Generate output file name: filename + sheetname
        base_name = os.path.splitext(excel_file)[0]
        output_file = f"{base_name}_{sheet_name}_deduplicated_M.xlsx"
        
        # Save merged results
        deduplicated_df.to_excel(output_file, index=False)
        
        print(f"✅ Found and merged {len(script_groups) - len(merged_data)} duplicate scripts")
        print(f"📁 Saved deduplicated scripts to: {output_file}")
        print(f"📊 Original points: {len(required_points_df)} -> Deduplicated points: {len(deduplicated_df)}")
        
        return deduplicated_df, True, output_file
    else:
        print("✅ No duplicate scripts found, proceeding with original data")
        return required_points_df, False, None

def analyze_script_uniqueness(required_points_df):
    """
    Analyze uniqueness at the Required_Discussion_Point level.
    - Aggregate all Standard_Script rows per point into a single point-level token set
    - Compute document frequency across points (not rows)
    - For each point, keep top 16 tokens by uniqueness score
    Returns point-specific weights structure: {point_name: {term: score}}
    """
    print("=== STEP 1: BUILD POINT-LEVEL DOCUMENTS ===")

    # Build point -> set of tokens aggregated across its script rows
    point_to_tokens = {}
    for idx, row in required_points_df.iterrows():
        script_text = row['Standard_Script']
        point_name = row['Required_Discussion_Point']
        if pd.isna(point_name) or pd.isna(script_text):
            continue
        tokens = set(tokenize_text(preprocess_text(script_text)))
        if point_name not in point_to_tokens:
            point_to_tokens[point_name] = set()
        point_to_tokens[point_name].update(tokens)

    # Compute document frequency across points
    print(f"Total points: {len(point_to_tokens)}")
    word_document_frequency = {}
    for point, tokens in point_to_tokens.items():
        for token in tokens:
            word_document_frequency[token] = word_document_frequency.get(token, 0) + 1

    print(f"Total unique tokens across points: {len(word_document_frequency)}")

    # Calculate uniqueness scores (IDF across points)
    total_documents = len(point_to_tokens)
    token_to_score = {}
    for token, df in word_document_frequency.items():
        idf_score = np.log(total_documents / df)
        business_boost = 2.0 if token in dictionaries.important_keywords else 1.0
        final_score = idf_score * business_boost
        token_to_score[token] = final_score

    print("=== STEP 2: SELECT TOP TERMS PER POINT (Point-Specific) ===")
    point_specific_weights = {}
    for point, tokens in point_to_tokens.items():
        # Rank tokens for this point by global uniqueness score
        scored = [(t, token_to_score.get(t, 0.0)) for t in tokens]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_16 = scored[:16]
        print(f"\nPoint: {point} | Selected {len(top_16)} terms")
        
        # Store point-specific weights
        point_weights = {}
        for t, s in top_16:
            print(f"  {t}: {s:.3f}")
            point_weights[t] = s
        
        point_specific_weights[point] = point_weights

    return point_specific_weights

# Function removed - now using external CSV.gz storage instead of writing back to dictionaries.py

def main():
    """Main function to run the dynamic script analysis."""
    parser = argparse.ArgumentParser(description='Dynamic Script Analysis Tool (Mandarin Version)')
    parser.add_argument('excel_file', nargs='?', default='call_text_sample.xlsx', 
                       help='Excel file containing scripts (default: call_text_sample.xlsx)')
    parser.add_argument('sheet_name', nargs='?', default='Script',
                       help='Sheet name containing scripts (default: Script)')
    parser.add_argument('--force', action='store_true',
                       help='Force recalculation even if scripts haven\'t changed')
    
    args = parser.parse_args()
    
    print("=== DYNAMIC SCRIPT ANALYSIS TOOL (MANDARIN VERSION) ===")
    print(f"Analyzing file: {args.excel_file}")
    print(f"Sheet name: {args.sheet_name}")
    
    # Check if file exists
    if not os.path.exists(args.excel_file):
        print(f"❌ Error: Excel file '{args.excel_file}' not found.")
        sys.exit(1)
    
    try:
        # Load script data
        xl = pd.ExcelFile(args.excel_file, engine='openpyxl')
        script_df = xl.parse(args.sheet_name)
        print(f"✅ Successfully loaded script data: {len(script_df)} rows")
    except Exception as e:
        print(f"❌ Error loading script data: {e}")
        sys.exit(1)
    
    print("🔄 Calculating dynamic term importance from scripts...")
    
    # Step 0: Check for and handle duplicate scripts
    deduplicated_df, duplicates_found, output_file = deduplicate_scripts(script_df, args.excel_file, args.sheet_name)
    
    if duplicates_found:
        print(f"📋 Using deduplicated script data for analysis")
        analysis_df = deduplicated_df
    else:
        analysis_df = script_df
    
    # Run analysis on deduplicated data - now returns point-specific structure
    point_specific_weights = analyze_script_uniqueness(analysis_df)
    
    # Detect product type for CSV export
    try:
        product_name = dictionaries.detect_product_type_from_script(analysis_df)
        if not product_name:
            product_name = "General"
    except:
        product_name = "General"
    
    print(f"\n=== EXPORTING TO CSV.GZ FORMAT ===")
    print(f"Product detected: {product_name}")
    
    # Import the universal CSV export function
    import sys
    sys.path.append('..')  # Add parent directory to path
    from dynamic_script_analysis import export_term_importance_to_csv
    
    # Construct the standard data structure for export
    all_results = {
        'MAN': {
            product_name: point_specific_weights
        }
    }
    
    # Export to CSV.gz using the universal exporter
    export_term_importance_to_csv(all_results, overwrite=True)
    
    # Calculate summary statistics
    total_points = len(point_specific_weights)
    total_weights = sum(len(weights) for weights in point_specific_weights.values())
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"✅ Generated weights for {total_points} discussion points")
    print(f"✅ Total term weights: {total_weights}")
    print(f"✅ Exported to CSV.gz format for external loading")
    print(f"✅ Data will be lazily loaded by dictionaries.py when needed")
    print(f"\nThe improved_call_coverage_checker_M.py will now use these externalized weights.")

if __name__ == "__main__":
    main()