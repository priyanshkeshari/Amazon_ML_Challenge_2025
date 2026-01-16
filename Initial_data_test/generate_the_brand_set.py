import pandas as pd
import re
from io import StringIO
import string

# --- 1. Load Data (Simulating the Training Data from your image) ---

# Note: In your actual environment, replace this with your actual file path:
df_train = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv')

# data_string = """
# sample_id,catalog_content,image_link,price                        
# 198967,"Salerno Cookies, The Original Butter Cookies, 8 Ounce (Pack of 4)",https://m.media-amazon.com/images/I/71XfHPR36-L.jpg,13.12
# 261251,"Bear Creek Hearty Soup Bowl, Creamy Chicken with Rice, 10 Ounce (Pack of 6)",https://m.media-amazon.com/images/I/51+FPE-wL.jpg,1.97
# 553558,"Judes&Art's Blue Cheese Powder, 11.25 oz - Gluten-Free and Nut-Free - Use in Seasonings and Salad Dressings - Great for Dips, Spreads and Sauces - Made in USA",https://m.media-amazon.com/images/I/41MdIH8Y0DL.jpg,30.34
# 293066,"Kederli Stemware Cherry Cooking Wine, 12.7 Ounce - 12 per case.",https://m.media-amazon.com/images/I/415A4D7F0ZL.jpg,66.49
# 9259,"Member's Mark Basil, 6.25 oz",https://m.media-amazon.com/images/I/81NCVHXh9RL.jpg,18.5
# 191345,"Goya Foods Sazonador Total Seasoning, 30 Ounce (Pack of 6)",https://m.media-amazon.com/images/I/61H0H2EHK0L.jpg,5.99
# 222007,"VineCo Original Series Chilean Sauvignon Blanc Wine Making Ingredient Kit",https://m.media-amazon.com/images/I/712IUaFonmL.jpg,94
# 37914,"NATURESPATH CEREAL FLY MULTIGRAIN ORE-ECO, 32 OZ, PK-6",https://m.media-amazon.com/images/I/21U99Rh2kL.jpg,35.74
# 230044,"Mrs. Miller's Seedless Black Raspberry Jam 9 Ounce (Pack of 4)",https://m.media-amazon.com/images/I/41Jm1K0k-l.jpg,31.9
# 254597,"Braswell's Key Lime Marmalade for Sale, 12oz",https://m.media-amazon.com/images/I/31S7QWbDrL.jpg,15.99
# 282323,"Albanese Gummy Bears, Sugar Free, 5-Pound Bags (Pack of 2)",https://m.media-amazon.com/images/I/71Cf7K7Qt5.jpg,33.5
# 21740,"KIZE Bars, 4 Count, Cookie Dough Flavor, High Protein, Dairy Free, Gluten Free, Non-GMO, Soy Free Snack Food Bars",https://m.media-amazon.com/images/I/41ASWW9RNAL.jpg,6.98
# 170292,"Smucker's Natural Peanut Butter Chunky, 16 OZ(Pack of 12)",https://m.media-amazon.com/images/I/C1XoRRYdl.jpg,4.255
# 71500,"BODYARMOR LITE Sports Drink-Low-Calorie Sports Beverage, Strawberry Banana, Coconut Water Hydration, Potassium-Packed Electrolytes, Perfect For Athletes, 20 Fl Oz (Pack of 12)",https://m.media-amazon.com/images/I/81U5RASA9gL.jpg,1.67
# 266475,"Organic Vinegar, Apple Cider",https://m.media-amazon.com/images/I/41SHfsFSL.jpg,81.44
# 86520,"Himalaya Pink Salt Fine Jar 10.0 Oz(Pack of 6)",https://m.media-amazon.com/images/I/514dNtht2L.jpg,56.72
# 125303,"BUSH'S BEST 16 oz Canned Barbecue Baked Beans, Source of Plant Based Protein and Fiber, Low Fat, Gluten Free, (Pack of 12)",https://m.media-amazon.com/images/I/71XfHPR36-L.jpg,5.47
# 271422,"BulkSupplements.com Trehalose Powder - Natural Sweetener, Trehalose Sugar Substitute - Vegan & Gluten Free, 5g per serving, 5kg (11 lbs) (Pack of 5)",https://m.media-amazon.com/images/I/61R5SASAuL.jpg,109.97
# 169842,"Soyarian Siberian All Day Set Instant Chaga Mushroom Tea, Pack of 3 Daily Tablets (90 Pcs) - Wild Harvested, Premium Blend, ""Natural Antioxidant Tea for Immune Support, Energy Boost & Improved",https://m.media-amazon.com/images/I/71YfK9CgL.jpg,50.99
# """
# df_train = pd.read_csv(StringIO(data_string.strip()))

# --- 2. CANDIDATE GENERATION FUNCTION ---

def extract_brand_candidate(content):
    """
    Extracts the most likely brand/product name candidate by splitting the string 
    at the first common separator (comma, hyphen, or specific unit).
    """
    # Look for the position of the first comma, opening parenthesis, or 'oz'/'ounce'
    separators = [',', '(', ' - ', ' oz', ' ounce', ' lbs']
    
    first_separator_index = len(content)
    
    # Find the index of the first encountered separator
    for sep in separators:
        index = content.lower().find(sep.lower())
        if index!= -1 and index < first_separator_index:
            first_separator_index = index
            
    candidate = content[:first_separator_index].strip()
    
    # Clean trailing punctuation
    candidate = candidate.rstrip(string.punctuation)
    
    # If the candidate is empty or too short, return the whole string (which usually indicates an unformatted title)
    return candidate if candidate else content.strip()

# Generate the initial list of brand candidates
brand_candidates = df_train['catalog_content'].apply(extract_brand_candidate).tolist()


# --- 3. REFINEMENT AND FILTERING ---

# Filter out common product types, generic descriptors, and uninformative entries.
# This list simulates the exclusion dictionary used in industrial NER systems.[1, 2]
GENERIC_FILTERS = [
    'tea', 'cookies', 'soup bowl', 'powder', 'wine', 'basil', 'seasoning', 
    'kit', 'cereal', 'jam', 'marmalade', 'gummy bears', 'bars', 'peanut butter',
    'sports drink', 'vinegar', 'pink salt', 'baked beans', 'powder', 'set',
    'chunky', 'natural', 'organic', 'seedless', 'original series', 'premium blend',
    'for sale', 'original'
]
# Create a set of unique candidates, lowercased for filtering
unique_candidates = set(c.lower() for c in brand_candidates)
final_brands = set()

for candidate in unique_candidates:
    is_generic = False
    
    # Check if the candidate is mostly a generic term
    for generic in GENERIC_FILTERS:
        if generic in candidate:
            # Check if the candidate contains multiple capitalized words (indicating a brand)
            if len(candidate.split()) <= 2 and candidate not in ('bushs best', 'goya foods'): 
                is_generic = True
                break
    
    # Add if it passes the filtering check and is not just a number/unit
    if not is_generic and not re.match(r'^\d', candidate):
        # We try to restore the original capitalization for the final list
        original_brand = [c for c in brand_candidates if c.lower() == candidate]
        if original_brand:
             # Add the best capitalized version found
             final_brands.add(original_brand)
        else:
             final_brands.add(candidate.title())

# --- 4. OUTPUT THE FINAL BRAND SET ---

# Convert back to a list, and ensure known multi-word brands are precisely represented
# Final manual cleanup of specific complex cases captured by the heuristic:
FINAL_BRAND_SET = sorted(list(final_brands))

# If the heuristic missed a known brand (like the full URL domain for BulkSupplements.com), add it back:
# KNOWN_EXACT_BRANDS = 

# for brand in KNOWN_EXACT_BRANDS:
#     if brand not in FINAL_BRAND_SET:
#         FINAL_BRAND_SET.append(brand)


print("--- Final Set of Extracted Brand Candidates for Feature Engineering ---")
# The resulting set can be used as a dictionary for lookups on both train and test data.
print(FINAL_BRAND_SET)