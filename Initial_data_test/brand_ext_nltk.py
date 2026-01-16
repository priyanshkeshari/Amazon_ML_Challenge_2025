import pandas as pd
import re
import string
from io import StringIO
import os
from nltk.corpus import stopwords
# Note: You must run 'import nltk; nltk.download('stopwords')' once before execution.

# --- 1. CONFIGURATION ---
OUTPUT_DIR = 'extracted_features'
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'extracted_brands.txt')

# --- 2. Load Data (Replace with your actual file loading) ---

# IMPORTANT: In your actual environment, replace this StringIO section with:
TRAIN_FILE_PATH = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv'
df_train = pd.read_csv(TRAIN_FILE_PATH)

# --- 3. BRAND CANDIDATE GENERATION FUNCTION (Regex + Punctuation) ---

def extract_brand_candidate_regex(content):
    """ Isolates the core 'Item Name' and extracts the leading phrase. """
    item_name_match = re.search(r"Item Name:\s*(.*?)(?:\n|$)", content, re.IGNORECASE)
    core_line = item_name_match.group(1).strip() if item_name_match else content.strip()
    
    separators = [',', ':', ' - ', ' oz', ' count', ' fl oz', ' lb']
    first_separator_index = len(core_line)
    
    for sep in separators:
        index = core_line.lower().find(sep) 
        if index!= -1 and index < first_separator_index:
            first_separator_index = index
            
    candidate = core_line[:first_separator_index].strip()
    candidate = candidate.rstrip(string.punctuation)
    
    return candidate if candidate else core_line.strip()

# Generate the initial list of brand candidates
brand_candidates = df_train['catalog_content'].apply(extract_brand_candidate_regex).tolist()


# --- 4. REFINEMENT AND FILTERING (NLTK Inspired Filtering) ---

STOP_WORDS = set(stopwords.words('english'))
DOMAIN_FILTERS = [
    'cookies', 'soup bowl', 'powder', 'wine', 'basil', 'seasoning', 'kit', 
    'cereal', 'jam', 'marinade', 'gummy bears', 'bars', 'peanut butter', 
    'sports drink', 'vinegar', 'salt', 'baked beans', 'tea', 'peas', 
    'sweetener', 'sugar substitute', 'sauce'
]

unique_candidates = set()

for candidate in brand_candidates:
    words = candidate.lower().split()
    if len(words) < 1: continue
    
    is_generic = False
    for word in words:
        if word in DOMAIN_FILTERS or word in STOP_WORDS:
            if len(words) <= 2: 
                is_generic = True
                break
    
    if re.match(r'^\d', candidate.strip()):
        is_generic = True

    if not is_generic:
        unique_candidates.add(candidate.strip())

FINAL_BRAND_SET = sorted(list(unique_candidates))


# --- 5. FILE OUTPUT AND EXECUTION SUMMARY ---

# 5.1 Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5.2 Write the final list of brands to the text file
try:
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# Extracted Brand Candidates from Training Data ({len(FINAL_BRAND_SET)} unique entries)\n")
        f.write(f"# Use this list to populate the KNOWN_BRANDS dictionary for feature engineering.\n\n")
        
        # Write each brand name on a new line
        for brand in FINAL_BRAND_SET:
            f.write(f"{brand}\n")

    print(f"\n--- SUCCESS: Brand List Saved ---")
    print(f"File created successfully at: {OUTPUT_FILE_PATH}")
    print(f"Total unique brands saved: {len(FINAL_BRAND_SET)}")

except Exception as e:
    print(f"Error writing file: {e}")