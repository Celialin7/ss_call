"""
============================================================
Dynamic Script Analysis Tool
============================================================

This standalone tool analyzes script files to generate dynamic term importance weights.
It exports weights to CSV.gz format for external loading by dictionaries.py.
No longer writes back to dictionaries.py (applies to both batch and single sheet modes).

Usage:
    python dynamic_script_analysis.py [excel_file] [sheet_name]

Configuration:
- USE_BATCH_MODE = True (default): Process all *_CAN/*_MAN sheets in Scripts.xlsx
- Paths configured via config.py: SCRIPT_FILE_PATH, TERM_IMPORTANCE_DIR, etc.

Features:
- Analyzes uniqueness of terms across all required discussion points
- Generates dynamic term importance weights based on rarity and theme relevance
- Keeps top 12 words with highest uniqueness score per discussion point
- Exports to CSV.gz format for lazy loading by dictionaries.py
- Supports multi-language processing (CAN/MAN/ENG)
- Always generates fresh results for each run

============================================================
"""

import pandas as pd
import numpy as np
import re
import pycantonese as pc
import jieba
import os
import sys
from collections import Counter

# Import dictionaries
import dictionaries

# Import configuration from config.py
try:
    from config import SCRIPT_FILE_PATH, SBERT_MODEL_PATH, SCRIPT_EMBEDDINGS_PATH
    print(f"✅ Using script file path from config.py: {SCRIPT_FILE_PATH}")
    SCRIPT_FILE_FULL_PATH = SCRIPT_FILE_PATH
except ImportError:
    print("⚠️  Warning: config.py not found, using fallback script file path")
    SCRIPT_FILE_FULL_PATH = "./Scripts.xlsx"

# SBERT imports for embedding generation
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: sentence-transformers not available. SBERT embedding generation will be skipped.")
    SBERT_AVAILABLE = False

import pickle

# Processing mode configuration
USE_BATCH_MODE = True  # Set to True for multi-product batch processing, False for single sheet mode
DEFAULT_SHEET_NAME = 'Script'  # Default sheet name for single sheet mode
DEFAULT_LANGUAGE = 'CAN'  # Default language for single sheet mode: 'CAN', 'MAN', or 'ENG'
# Note: Analysis always runs fresh - no caching

def sanitize_string_for_dict_key(text):
    """
    安全字符串化函数：处理讨论点名称中的换行符和特殊字符，确保生成合法的Python字典键。
    
    处理两个主要问题：
    1. 去除所有换行符（\r、\n、\r\n），避免跨行破坏字典语法
    2. 转义单引号和反斜杠，避免字面量语法错误
    
    Args:
        text: 原始字符串（可能包含换行符、单引号等）
        
    Returns:
        str: 安全的字符串，可用作Python字典键
    """
    if pd.isna(text):
        return ""
    
    # 转换为字符串
    safe_text = str(text)
    
    # 步骤1: 去除所有类型的换行符
    safe_text = safe_text.replace('\r\n', ' ')  # Windows换行符
    safe_text = safe_text.replace('\r', ' ')    # Mac经典换行符
    safe_text = safe_text.replace('\n', ' ')    # Unix换行符
    
    # 步骤2: 转义单引号（避免字面量语法错误）
    safe_text = safe_text.replace("'", "\\'")
    
    # 步骤3: 转义反斜杠（避免转义序列问题）
    safe_text = safe_text.replace("\\", "\\\\")
    
    # 步骤4: 去除首尾空白，压缩内部多余空格
    safe_text = re.sub(r'\s+', ' ', safe_text.strip())
    
    return safe_text

def preprocess_text(text):
    """Enhanced text preprocessing for Cantonese and speech-to-text errors.
    
    Note: Removes English letters to prevent them from being treated as keywords
    in term_importance analysis, focusing only on Chinese business terms.
    """
    # Keep Chinese characters (simplified/traditional) and specified punctuation
    # Remove English letters and all other characters including spaces, newlines, tabs, etc.
    text = re.sub(r'[^\u4e00-\u9fff,.。，。%()（）]+', '', str(text))
    # Normalize common speech-to-text errors
    for correct, variations in dictionaries.error_patterns.items():
        for variation in variations:
            text = text.replace(variation, correct)
    return text.strip()

def tokenize_text(text, language='CAN', remove_stopwords=True):
    """
    Multi-language tokenization with language-specific stopword removal.
    
    Args:
        text: Input text to tokenize
        language: 'CAN' for Cantonese, 'MAN' for Mandarin, 'ENG' for English
        remove_stopwords: Always True for dynamic analysis to ensure high-quality keywords
    
    Returns:
        List of tokens, filtered with appropriate stopwords for the language
    """
    if language == 'CAN':
        # Cantonese: Use pycantonese segmentation
        tokens = pc.segment(text)
    elif language in ['MAN', 'ENG']:
        # Mandarin/English: Use jieba segmentation
        tokens = jieba.lcut(text)
    else:
        # Fallback: Use jieba for unknown languages
        tokens = jieba.lcut(text)
    
    # Always remove stopwords for dynamic analysis to ensure keyword quality
    if remove_stopwords:
        # Get language-specific stopwords
        current_stopwords = dictionaries.get_stopwords(language)
        # Remove stopwords and short tokens
        business_tokens = [token for token in tokens if token not in current_stopwords and len(token) > 1]
    else:
        # Keep more tokens, only remove very short ones
        business_tokens = [token for token in tokens if len(token) > 1]
    
    return business_tokens

def parse_script_variations(script_text):
    """
    Parse script text to extract all variations for SBERT embedding generation.
    This function is needed for Mandarin SBERT preprocessing.
    
    Args:
        script_text: Script text that may contain multiple variations
        
    Returns:
        list: All script variations as separate sentences
    """
    if pd.isna(script_text):
        return []
    
    # Simple splitting by common delimiters
    # You can enhance this based on your script format
    variations = []
    
    # Split by common separators
    for separator in ['。', '；', '\n', '|']:
        if separator in script_text:
            parts = script_text.split(separator)
            for part in parts:
                part = part.strip()
                if part and len(part) > 3:  # Filter out very short fragments
                    variations.append(part)
            return variations
    
    # If no separators found, return the whole text as one variation
    script_text = script_text.strip()
    if script_text:
        variations.append(script_text)
    
    return variations

def generate_script_embeddings(all_scripts_df, model, output_path):
    """
    Generate and save SBERT embeddings for all script variations.
    
    Args:
        all_scripts_df: DataFrame containing all Mandarin scripts with Standard_Script column
        model: Loaded SentenceTransformer model
        output_path: Path to save the embeddings pickle file
        
    Returns:
        dict: Dictionary mapping script sentences to their embeddings
    """
    print("=== GENERATING SBERT SCRIPT EMBEDDINGS ===")
    
    if not SBERT_AVAILABLE:
        print("❌ SBERT not available, skipping embedding generation")
        return {}
    
    # Extract all unique Standard_Script texts
    unique_scripts = set()
    for _, row in all_scripts_df.iterrows():
        script_text = row.get('Standard_Script', '')
        if pd.notna(script_text) and script_text.strip():
            unique_scripts.add(script_text.strip())
    
    print(f"📊 Found {len(unique_scripts)} unique scripts")
    
    # Parse all script variations and create mapping from original to preprocessed
    original_to_preprocessed = {}
    unique_preprocessed_sentences = set()
    
    for script_text in unique_scripts:
        # Preprocess full script to remove English, keep only Chinese
        preprocessed_script = preprocess_text(script_text)
        if preprocessed_script:  # Only add if not empty after preprocessing
            original_to_preprocessed[script_text] = preprocessed_script
            unique_preprocessed_sentences.add(preprocessed_script)
        
        # Add all variations from parse_script_variations
        variations = parse_script_variations(script_text)
        for variation in variations:
            if variation and variation.strip():
                # Preprocess each variation to remove English
                preprocessed_variation = preprocess_text(variation.strip())
                if preprocessed_variation:  # Only add if not empty after preprocessing
                    original_to_preprocessed[variation.strip()] = preprocessed_variation
                    unique_preprocessed_sentences.add(preprocessed_variation)
    
    # Convert to list for encoding (only unique preprocessed sentences)
    preprocessed_sentences_list = list(unique_preprocessed_sentences)
    print(f"📊 Total unique preprocessed script sentences: {len(preprocessed_sentences_list)}")
    
    # Batch encode all preprocessed script sentences
    print("🔄 Encoding preprocessed script sentences with SBERT...")
    try:
        script_embeddings = model.encode(preprocessed_sentences_list, show_progress_bar=True)
        print(f"✅ Successfully encoded {len(script_embeddings)} preprocessed script sentences")
    except Exception as e:
        print(f"❌ Error during SBERT encoding: {e}")
        return {}
    
    # Create preprocessed to embedding mapping
    preprocessed_to_embedding = {}
    for sentence, embedding in zip(preprocessed_sentences_list, script_embeddings):
        preprocessed_to_embedding[sentence] = embedding
    
    # Create final embeddings dictionary: original text -> embedding of preprocessed text
    embeddings_dict = {}
    for original_text, preprocessed_text in original_to_preprocessed.items():
        if preprocessed_text in preprocessed_to_embedding:
            embeddings_dict[original_text] = preprocessed_to_embedding[preprocessed_text]
    
    # Save embeddings to pickle file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"✅ Saved script embeddings to: {output_path}")
        print(f"   📊 Embeddings saved: {len(embeddings_dict)}")
    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
        return {}
    
    return embeddings_dict

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
        output_file = f"{base_name}_{sheet_name}_deduplicated.xlsx"
        
        # Save merged results
        deduplicated_df.to_excel(output_file, index=False)
        
        print(f"✅ Found and merged {len(script_groups) - len(merged_data)} duplicate scripts")
        print(f"📁 Saved deduplicated scripts to: {output_file}")
        print(f"📊 Original points: {len(required_points_df)} -> Deduplicated points: {len(deduplicated_df)}")
        
        return deduplicated_df, True, output_file
    else:
        print("✅ No duplicate scripts found, proceeding with original data")
        return required_points_df, False, None

def analyze_script_uniqueness(required_points_df, language='CAN'):
    """
    Analyze uniqueness at the Required_Discussion_Point level with language-specific processing.
    - Aggregate all Standard_Script rows per point into a single point-level token set
    - Compute document frequency across points (not rows)
    - For each point, keep top 12 tokens by uniqueness score
    - Use appropriate tokenization strategy based on target language and model
    
    Args:
        required_points_df: DataFrame with Required_Discussion_Point and Standard_Script columns
        language: 'CAN' for Cantonese (TF-IDF mode), 'MAN'/'ENG' for Mandarin/English (SBERT mode)
    
    Returns:
        dict: Merged dynamic weights across all points
    """
    print(f"=== STEP 1: BUILD POINT-LEVEL DOCUMENTS ({language} MODE) ===")

    # Dynamic analysis always removes stopwords for all languages to ensure keyword quality
    print(f"🎯 Dynamic Analysis Mode: Removing {language} stopwords for optimal keyword precision")
    print("   This ensures term_importance contains only high-quality, distinctive business terms")

    # Build point -> set of tokens aggregated across its script rows
    point_to_tokens = {}
    for idx, row in required_points_df.iterrows():
        script_text = row['Standard_Script']
        point_name = row['Required_Discussion_Point']
        if pd.isna(point_name) or pd.isna(script_text):
            continue
        # Always remove stopwords for dynamic analysis (regardless of final model usage)
        tokens = set(tokenize_text(preprocess_text(script_text), language=language, remove_stopwords=True))
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
        top_12 = scored[:12]
        print(f"\nPoint: {point} | Selected {len(top_12)} terms")
        
        # Store point-specific weights
        point_weights = {}
        for t, s in top_12:
            print(f"  {t}: {s:.3f}")
            point_weights[t] = s
        
        # Final safety: remove any leaked stopwords for this point
        try:
            language_stopwords = dictionaries.get_stopwords(language)
            leaked = [w for w in list(point_weights.keys()) if w in language_stopwords]
            if leaked:
                print(f"    ⚠️  Safety filter: Removing {len(leaked)} stopwords for {point} → {leaked}")
                for w in leaked:
                    point_weights.pop(w, None)
        except Exception as e:
            print(f"    ⚠️  Safety stopword filter error for {point}: {e}")
        
        point_specific_weights[point] = point_weights

    return point_specific_weights

# Function removed - now using external CSV.gz storage instead of writing back to dictionaries.py

def export_term_importance_to_csv(all_results, overwrite=True):
    """
    Export term importance data to CSV.gz format for external storage.
    
    Args:
        all_results: dict like {'CAN': {'Bond': {...}, 'Fund': {...}}, 'MAN': {...}}
        overwrite: If True, delete existing file before writing
    """
    print(f"📊 Exporting term importance data to CSV.gz format...")
    
    # Import config for paths
    try:
        from config import TERM_IMPORTANCE_DIR, TERM_IMPORTANCE_CSV
        csv_path = os.path.join(TERM_IMPORTANCE_DIR, TERM_IMPORTANCE_CSV)
    except ImportError:
        # Fallback paths if config import fails
        csv_path = os.path.join("Generated/SS_project", "term_importance.csv.gz")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Remove existing file if overwrite is True
    if overwrite and os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"🗑️  Removed existing file: {csv_path}")
    
    # Build DataFrame from all_results
    rows = []
    for language in ['CAN', 'MAN', 'ENG']:
        if language in all_results and all_results[language]:
            for product_name, point_specific_weights in all_results[language].items():
                for point_name, point_weights in point_specific_weights.items():
                    # Safety: filter out language-specific stopwords
                    try:
                        language_stopwords = dictionaries.get_stopwords(language)
                        clean_weights = {w: s for w, s in point_weights.items() 
                                       if w not in language_stopwords and w and w.strip()}
                    except Exception:
                        clean_weights = point_weights
                    
                    # Add each term as a row
                    for term, weight in clean_weights.items():
                        rows.append({
                            'language': language,
                            'product': product_name,
                            'point': point_name,
                            'term': term,
                            'weight': weight
                        })
    
    # Create DataFrame and export
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, compression='gzip', index=False)
        
        print(f"✅ Exported {len(rows)} term importance records to: {csv_path}")
        print(f"   - Languages: {len(df['language'].unique())}")
        print(f"   - Products: {len(df['product'].unique())}")
        print(f"   - Points: {len(df['point'].unique())}")
        print(f"   - Unique terms: {len(df['term'].unique())}")
    else:
        print("⚠️  No data to export")

def update_dictionaries_with_products(all_results):
    """
    Export term importance data to CSV.gz and update dictionaries.py structure.
    This replaces the old approach of storing large data directly in dictionaries.py.
    
    Args:
        all_results: dict like {'CAN': {'Bond': {...}, 'Fund': {...}}, 'MAN': {...}}
    """
    print(f"🔄 Externalizing term importance data to CSV.gz format...")
    
    # Step 1: Export data to CSV.gz
    export_term_importance_to_csv(all_results, overwrite=True)
    
    # Step 2: Update dictionaries.py to remove large data constants (if they exist)
    # This will be handled in the next phase when we refactor dictionaries.py
    
    # Show summary
    total_languages = len([k for k, v in all_results.items() if v])
    total_products = sum(len(products) for products in all_results.values())
    total_points = sum(len(points) for products in all_results.values() for points in products.values())
    total_weights = sum(len(weights) for products in all_results.values() 
                       for points in products.values() for weights in points.values())
    
    print(f"✅ Term importance data externalization complete:")
    print(f"   - Languages: {total_languages}")
    print(f"   - Products: {total_products}")
    print(f"   - Discussion points: {total_points}")
    print(f"   - Total term weights: {total_weights}")
    print(f"   - Data stored externally in CSV.gz format")

def extract_product_info(sheet_name):
    """
    Extract product and language info from sheet name
    Examples:
    - "Bond_CAN" -> ("Bond", "CAN")
    - "Caller Linear Note_MAN" -> ("Caller Linear Note", "MAN")
    """
    if '_' not in sheet_name:
        return None, None
    
    parts = sheet_name.rsplit('_', 1)  # Split from right, only once
    product_name = parts[0]
    language = parts[1] if len(parts) == 2 and parts[1] in ['CAN', 'MAN', 'ENG'] else None
    
    return product_name, language

def process_all_products_batch(excel_file):
    """
    Batch process all products in Scripts.xlsx
    
    Process:
    1. Read all sheets
    2. Filter sheets matching *_CAN, *_MAN format
    3. Run deduplication + dynamic analysis for each sheet
    4. Organize results by language and product
    5. Update dictionaries.py
    """
    print("=== BATCH MODE: PROCESSING ALL PRODUCTS ===")
    
    if not os.path.exists(excel_file):
        print(f"❌ Error: Excel file '{excel_file}' not found.")
        sys.exit(1)
    
    try:
        xl = pd.ExcelFile(excel_file, engine='openpyxl')
        all_sheets = xl.sheet_names
        print(f"📊 Found {len(all_sheets)} sheets in {excel_file}")
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        sys.exit(1)
    
    # Filter product sheets
    product_sheets = []
    for sheet_name in all_sheets:
        product_name, language = extract_product_info(sheet_name)
        if product_name and language:
            product_sheets.append((sheet_name, product_name, language))
            print(f"✅ Found product sheet: {sheet_name} -> {language}:{product_name}")
    
    if not product_sheets:
        print("❌ No product sheets found with format ProductName_CAN/MAN/ENG")
        sys.exit(1)
    
    print(f"\n🔄 Processing {len(product_sheets)} product sheets...")
    
    # Organize result structure
    all_results = {
        'CAN': {},
        'MAN': {},
        'ENG': {}
    }
    
    # Process each product sheet
    for sheet_name, product_name, language in product_sheets:
        print(f"\n--- Processing {language}:{product_name} ({sheet_name}) ---")
        
        try:
            # Load data
            script_df = xl.parse(sheet_name)
            print(f"📊 Loaded {len(script_df)} rows from {sheet_name}")
            
            # Deduplication process
            deduplicated_df, duplicates_found, output_file = deduplicate_scripts(
                script_df, excel_file, sheet_name
            )
            
            analysis_df = deduplicated_df if duplicates_found else script_df
            
            # Dynamic analysis with language-specific processing
            point_specific_weights = analyze_script_uniqueness(analysis_df, language=language)
            
            # Store results
            all_results[language][product_name] = point_specific_weights
            total_points = len(point_specific_weights)
            total_weights = sum(len(weights) for weights in point_specific_weights.values())
            print(f"✅ Generated {total_weights} weights across {total_points} points for {language}:{product_name}")
            
        except Exception as e:
            print(f"❌ Error processing {sheet_name}: {e}")
            continue
    
    # Update dictionaries.py
    print(f"\n=== UPDATING DICTIONARIES WITH ALL RESULTS ===")
    update_dictionaries_with_products(all_results)
    
    # Show summary
    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    total_products = sum(len(products) for products in all_results.values())
    print(f"✅ Processed {total_products} products across {len([k for k, v in all_results.items() if v])} languages")
    
    for language, products in all_results.items():
        if products:
            print(f"📊 {language}: {list(products.keys())}")
    
    # Generate SBERT embeddings for Mandarin scripts
    if SBERT_AVAILABLE and 'MAN' in all_results and all_results['MAN']:
        print(f"\n=== GENERATING SBERT EMBEDDINGS FOR MANDARIN SCRIPTS ===")
        
        try:
            # Import config for paths
            import sys
            sys.path.append('.')
            from config import SBERT_MODEL_PATH
            
            # Load SBERT model
            print(f"🔄 Loading SBERT model from: {SBERT_MODEL_PATH}")
            sbert_model = SentenceTransformer(SBERT_MODEL_PATH)
            print("✅ SBERT model loaded successfully")
            
            # Collect all Mandarin scripts
            all_mandarin_scripts = []
            for sheet_name, product_name, language in product_sheets:
                if language == 'MAN':
                    try:
                        script_df = xl.parse(sheet_name)
                        all_mandarin_scripts.append(script_df)
                        print(f"📊 Added {len(script_df)} scripts from {sheet_name}")
                    except Exception as e:
                        print(f"⚠️  Error loading {sheet_name} for embeddings: {e}")
            
            if all_mandarin_scripts:
                # Combine all Mandarin scripts
                combined_mandarin_df = pd.concat(all_mandarin_scripts, ignore_index=True)
                print(f"📊 Combined {len(combined_mandarin_df)} total Mandarin script rows")
                
                # Generate embeddings  
                from config import SCRIPT_EMBEDDINGS_PATH
                embedding_output_path = SCRIPT_EMBEDDINGS_PATH
                embeddings_dict = generate_script_embeddings(
                    combined_mandarin_df, 
                    sbert_model, 
                    embedding_output_path
                )
                
                if embeddings_dict:
                    print(f"✅ SBERT embeddings generation completed successfully")
                    print(f"   📁 Embeddings file: {embedding_output_path}")
                    print(f"   📊 Total embeddings: {len(embeddings_dict)}")
                else:
                    print("❌ SBERT embeddings generation failed")
            else:
                print("⚠️  No Mandarin scripts found for embedding generation")
                
        except Exception as e:
            print(f"❌ Error during SBERT embedding generation: {e}")
            print("   Term importance analysis completed successfully, but embeddings generation failed")
    else:
        if not SBERT_AVAILABLE:
            print("⚠️  SBERT not available, skipping embedding generation")
        elif 'MAN' not in all_results or not all_results['MAN']:
            print("⚠️  No Mandarin scripts processed, skipping embedding generation")

def main():
    """Main function to run the dynamic script analysis."""
    
    if USE_BATCH_MODE:
        # Batch processing mode
        process_all_products_batch(SCRIPT_FILE_FULL_PATH)
    else:
        # Single sheet processing mode (backward compatibility)
        print("=== SINGLE SHEET MODE ===")
        print(f"Analyzing file: {SCRIPT_FILE_FULL_PATH}")
        print(f"Sheet name: {DEFAULT_SHEET_NAME}")
        
        # Check if file exists
        if not os.path.exists(SCRIPT_FILE_FULL_PATH):
            print(f"❌ Error: Excel file '{SCRIPT_FILE_FULL_PATH}' not found.")
            sys.exit(1)
        
        try:
            # Load script data
            xl = pd.ExcelFile(SCRIPT_FILE_FULL_PATH, engine='openpyxl')
            script_df = xl.parse(DEFAULT_SHEET_NAME)
            print(f"✅ Successfully loaded script data: {len(script_df)} rows")
        except Exception as e:
            print(f"❌ Error loading script data: {e}")
            sys.exit(1)
        
        print("🔄 Calculating dynamic term importance from scripts...")
        
        # Step 0: Check for and handle duplicate scripts
        deduplicated_df, duplicates_found, output_file = deduplicate_scripts(script_df, SCRIPT_FILE_FULL_PATH, DEFAULT_SHEET_NAME)
        
        if duplicates_found:
            print(f"📋 Using deduplicated script data for analysis")
            analysis_df = deduplicated_df
        else:
            analysis_df = script_df
        
        # Run analysis on deduplicated data with language-specific processing
        point_specific_weights = analyze_script_uniqueness(analysis_df, language=DEFAULT_LANGUAGE)

        print(f"\n=== FINAL RESULTS ===")
        total_points = len(point_specific_weights)
        total_weights = sum(len(weights) for weights in point_specific_weights.values())
        print(f"Total dynamic weights generated: {total_weights} across {total_points} points")

        # Show all point-specific weights
        print(f"\nPoint-Specific Dynamic Weights:")
        for point_name, point_weights in point_specific_weights.items():
            print(f"\n  📍 {point_name} ({len(point_weights)} weights):")
            sorted_weights = sorted(point_weights.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_weights:
                print(f"    {word}: {score:.3f}")

        # For single sheet mode, create a flattened version for backward compatibility
        flattened_weights = {}
        for point_weights in point_specific_weights.values():
            for word, score in point_weights.items():
                # Keep highest score if word appears in multiple points
                if word not in flattened_weights or score > flattened_weights[word]:
                    flattened_weights[word] = score

        # Create structure for the new system (single sheet = single product)
        single_sheet_results = {
            DEFAULT_LANGUAGE: {
                'SingleSheet': point_specific_weights  # Store point-specific weights
            }
        }

        # Export to CSV.gz format for external loading
        update_dictionaries_with_products(single_sheet_results)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"✅ Generated {total_weights} dynamic weights across {total_points} points")
        print(f"✅ Exported to CSV.gz format for external loading")
        print(f"\nThe improved_call_coverage_checker.py will now use these externalized weights.")

if __name__ == "__main__":
    main()
