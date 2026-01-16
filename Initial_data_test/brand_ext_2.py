import pandas as pd
import re
import string
import os
from io import StringIO
from nltk.corpus import stopwords
# Note: You must run 'import nltk; nltk.download('stopwords')' once before execution.

# --- 1. CONFIGURATION AND DIRECTORY SETUP ---
OUTPUT_DIR = 'extracted_features'
BRAND_FILE_NAME = 'refined_brands.txt'
BRAND_FILE_PATH = os.path.join(OUTPUT_DIR, BRAND_FILE_NAME)

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' ensured.")

# --- 2. Load Data (Replace with your actual file loading) ---
# IMPORTANT: Use the path to your full 75,000-row train.csv here.
TRAIN_FILE_PATH = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv'
df_train = pd.read_csv(TRAIN_FILE_PATH)


# --- 3. ENHANCED BRAND CANDIDATE EXTRACTION ---

# Custom domain-specific filters expanded to handle generic prefixes/suffixes
DOMAIN_FILTERS = [
    'cookies', 'soup bowl', 'powder', 'wine', 'basil', 'seasoning', 'kit', 
    'cereal', 'jam', 'marinade', 'gummy bears', 'bars', 'peanut butter', 
    'sports drink', 'vinegar', 'salt', 'baked beans', 'tea', 'peas', 
    'sweetener', 'sugar substitute', 'sauce', 'organic', 'natural', 'fresh',
    'made in usa' # Common suffix that is not a brand
]
STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = str.maketrans('', '', string.punctuation)


def extract_robust_brand(content):
    """
    Pass 1: Use Regex to isolate the core 'Item Name' and extract the leading phrase.
    """
    item_name_match = re.search(r"Item Name:\s*(.*?)(?:\n|$)", content, re.IGNORECASE)
    core_line = item_name_match.group(1).strip() if item_name_match else content.strip()
    
    # Target common separators that follow a brand (comma, size units, brackets)
    separators = [',', ':', ' - ', ' oz', ' count', ' fl oz', ' lb', '(', '[']
    first_separator_index = len(core_line)
    
    for sep in separators:
        index = core_line.lower().find(sep) 
        if index!= -1 and index < first_separator_index:
            first_separator_index = index
            
    candidate = core_line[:first_separator_index].strip()
    candidate = candidate.rstrip(string.punctuation)
    
    if not candidate:
        return "Unknown_Empty" # Safety return

    """
    Pass 2: Token-Level Validation and Filtering (High Precision Check)
    This checks capitalization and ensures the candidate is not a generic product name.
    """
    
    # Split the candidate into tokens for analysis
    # Remove punctuation for word-level checks
    words = candidate.translate(PUNCTUATION).split()
    
    # Heuristic 1: If the candidate is 4 words or less, check if it's primarily composed of 
    # capitalized words and is not a generic term (simulating NER feature).
    if len(words) <= 4:
        # Check capitalization: at least 50% of tokens should be title-cased or uppercase
        capitalized_count = sum(word.istitle() or word.isupper() for word in words if len(word) > 1)
        
        # Check against filters
        is_generic = any(word.lower() in DOMAIN_FILTERS or word.lower() in STOP_WORDS for word in words)
        
        # Accept if it's highly capitalized AND not primarily generic
        if capitalized_count >= len(words) / 2 and not is_generic:
            return candidate.strip()
    
    # If it fails the capitalization/length check, it's likely a generic description that was accidentally parsed first.
    
    # Heuristic 2: Final clean check against known domain filters (e.g., catching "Organic Vinegar")
    if any(candidate.lower().endswith(f) for f in DOMAIN_FILTERS):
        return "Unknown_Filtered"
        
    # Final acceptance: If the candidate contains multiple words and is clearly capitalized.
    if len(words) > 1 and candidate.istitle():
        return candidate.strip()
    
    return "Unknown_Generic"

# --- 4. EXECUTION AND SAVING ---

# Apply the robust extraction function
df_train['extracted_brand_raw'] = df_train['catalog_content'].apply(extract_robust_brand)

# Filter the final, unique, non-unknown brands
FINAL_BRAND_SET = sorted(df_train[~df_train['extracted_brand_raw'].str.startswith('Unknown')]['extracted_brand_raw'].unique())


# --- 5. FILE OUTPUT AND EXECUTION SUMMARY ---

try:
    with open(BRAND_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# Extracted Brand Candidates (Enhanced Heuristic) from Training Data\n")
        f.write(f"# Total unique brands identified: {len(FINAL_BRAND_SET)}\n")
        f.write(f"# Use this list to populate the KNOWN_BRANDS dictionary for One-Hot Encoding.\n\n")
        
        for brand in FINAL_BRAND_SET:
            f.write(f"{brand}\n")

    print(f"\n--- SUCCESS: Refined Brand List Saved ---")
    print(f"File created successfully at: {BRAND_FILE_PATH}")
    print(f"Total unique brands saved: {len(FINAL_BRAND_SET)}")
    
    print("\n--- Sample of Results on Provided Data ---")
    print(df_train[['catalog_content', 'extracted_brand_raw']].head(8))
    
except Exception as e:
    print(f"Error writing file: {e}")