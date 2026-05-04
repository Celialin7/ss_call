# Configuration file for batch analysis system

# Path configurations
CONVERTED_TEXT_FOLDER = "./converted_text"  # Folder containing .csv files to process
FILE_MAPPING_PATH = "./file_mapping.xlsx"  # Excel file mapping Sample No to Product Name
SCRIPT_FILE_PATH = "./Scripts.xlsx"  # Script file path
OUTPUT_FOLDER = "./output"  # Folder to save all output files
LOG_FILE_PATH = "./batch_analysis.log"  # Log file path

# SBERT model path for Mandarin version
SBERT_MODEL_PATH = "./multilingual_sbert/paraphrase-multilingual-MiniLM-L12-v2"

# SBERT embeddings path for precomputed script vectors
SCRIPT_EMBEDDINGS_PATH = "./script_embeddings.pkl"

# System audio recording inclusion control
INCLUDE_SYSTEM_AUDIO_IN_ANALYSIS = False  # Set to True to include system recordings in analysis

# Term importance data storage configuration
TERM_IMPORTANCE_DIR = "Generated/SS_project"  # Directory for term importance data
TERM_IMPORTANCE_CSV = "term_importance.csv.gz"  # Compressed CSV file for term importance data

# ==========================================================
# Script Point Grouping Configuration
# ==========================================================
# For listed sheets only, Required_Discussion_Point will be combined with
# "Points to Note" as grouping key:
#   "<Required_Discussion_Point> | <Points to Note>"
#
# Example:
# COMPOSITE_POINT_KEY_SHEETS = [
#     "NewProduct_CAN",
#     "NewProduct_MAN",
# ]
COMPOSITE_POINT_KEY_SHEETS = []
COMPOSITE_POINT_KEY_SEPARATOR = " | "
COMPOSITE_POINT_NOTE_COLUMN = "Points to Note"


def should_use_composite_point_key(sheet_name):
    """
    Whether the given script sheet should use the composite grouping key:
    Required_Discussion_Point + Points to Note.
    """
    if not sheet_name:
        return False
    return str(sheet_name).strip() in {str(s).strip() for s in COMPOSITE_POINT_KEY_SHEETS}

# Language-specific similarity weights configuration
LANGUAGE_SIMILARITY_WEIGHTS = {
    'MAN': {
        'SBERT': 0.60,
        'expanded_overlap': 0.15,
        'rouge_l': 0.15,
        'keyword_coverage': 0.10
    },
    'CAN': {
        'SBERT': 0.60,
        'expanded_overlap': 0.15,
        'rouge_l': 0.15,
        'keyword_coverage': 0.10
    },
    'ENG': {
        'SBERT': 0.60,
        'expanded_overlap': 0.13,
        'rouge_l': 0.12,
        'keyword_coverage': 0.10,
        'fuzzy_similarity': 0.05
    }
}

def get_similarity_weights(language):
    """
    Get similarity weights for specified language.
    
    Args:
        language: Language code ('MAN', 'CAN', 'ENG')
        
    Returns:
        dict: Similarity weights for the language, defaults to MAN if not found
    """
    return LANGUAGE_SIMILARITY_WEIGHTS.get(language, LANGUAGE_SIMILARITY_WEIGHTS['MAN'])
