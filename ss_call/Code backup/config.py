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
