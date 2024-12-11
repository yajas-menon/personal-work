import asyncio
import os
import logging
import json
from typing import List, Dict, Optional
import pandas as pd
from pydantic import BaseModel
import litellm
from dotenv import load_dotenv
import Levenshtein
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AddressEvent(BaseModel):
    area: str
    emirates: str

def normalize_string(text: str) -> str:
    """Normalize string for comparison"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

def normalize_and_tokenize(text: str) -> set:
    """Normalize text and return set of words"""
    if not isinstance(text, str):
        return set()
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    return set(text.split())

def create_temp_result(reason: str) -> Dict[str, str]:
    """Helper function to create TEMP result"""
    return {
        'area': 'TEMP',
        'emirates': '',
        'zone': '',
        'similarity_score': 0,
        'temp_reason': reason
    }

def match_with_master(llm_result: Dict[str, str], full_address: str, master_df: pd.DataFrame) -> Dict[str, str]:
    """
    Match addresses using a clear step-by-step logic:
    1. Try exact match with LLM area.
    2. If not found, look for master areas in address.
    3. Validate substring matches with proper boundaries.
    4. Mark as TEMP if no matches are found.
    """
    try:
        llm_area = normalize_string(llm_result.get('area', ''))
        llm_emirate = normalize_string(llm_result.get('emirates', ''))
        normalized_address = normalize_string(full_address)
        
        # Step 1: Basic validation
        if not llm_area:
            temp_reason = "LLM did not identify any area"
            logger.info(f"Address marked TEMP: {temp_reason}")
            return create_temp_result(temp_reason)

        logger.info(f"Processing area: {llm_result['area']} for full address: {full_address}")
        
        # Step 2: Try exact match with LLM identified area
        exact_matches = master_df[master_df['Area'].str.lower() == llm_area]
        if not exact_matches.empty:
            area_row = exact_matches.iloc[0]
            
            # Validate area is not the same as emirate
            if normalize_string(area_row['Area']) == normalize_string(area_row['Emirate']):
                temp_reason = f"Area and Emirate cannot be the same ({area_row['Area']})"
                logger.info(f"Address marked TEMP: {temp_reason}")
                return create_temp_result(temp_reason)
                
            logger.info(f"Exact match found: {area_row['Area']}")
            return {
                'area': area_row['Area'],
                'emirates': area_row['Emirate'],
                'zone': area_row['Zone'],
                'similarity_score': 1.0,
                'temp_reason': '',
                'row_index': exact_matches.index[0] + 2  # Adjust for row index
            }
            
        # Step 3: Search for master areas in address using stricter substring matching
        logger.info(f"No exact match found for {llm_area}, searching address for master areas")
        
        found_matches = []
        for idx, row in master_df.iterrows():
            area_normalized = normalize_string(row['Area'])
            # Use stricter substring matching with word boundaries
            if re.search(rf'\b{re.escape(area_normalized)}\b', normalized_address):
                # Validate emirate if available
                if llm_emirate and normalize_string(row['Emirate']) == llm_emirate:
                    score = 0.95  # Higher score for emirate match
                else:
                    score = 0.9
                    
                found_matches.append({
                    'area': row['Area'],
                    'emirates': row['Emirate'],
                    'zone': row['Zone'],
                    'similarity_score': score,
                    'temp_reason': '',
                    'length': len(area_normalized.split()),  # Prefer longer area names
                    'row_index': idx + 2  # Adjust for row index
                })
        
        # Step 4: If master areas found in address, return best match
        if found_matches:
            # Sort by score and length to get best match
            best_match = max(found_matches, key=lambda x: (x['similarity_score'], x['length']))
            logger.info(f"Found master area in address: {best_match['area']}")
            return best_match
            
        # Step 5: No matches found, mark as TEMP
        temp_reason = f"No matching area found in master data for: {full_address}"
        logger.info(f"Address marked TEMP: {temp_reason}")
        return create_temp_result(temp_reason)
        
    except Exception as e:
        temp_reason = f"Error in matching process: {str(e)}"
        logger.error(f"Address marked TEMP: {temp_reason}")
        return create_temp_result(temp_reason)


async def test_get_response(location: str) -> Dict[str, str]:
    """Get area and emirate from LLM"""
    user_message = f"""As an address analysis expert, carefully analyze the given address to extract the AREA and EMIRATE. Follow these guidelines:

TASK:
- Extract the main AREA name (neighborhood/district/community)
- Identify the EMIRATE (Dubai, Abu Dhabi, Sharjah, Ajman, Ras Al Khaimah, Fujairah, Umm Al Quwain)

RULES:
1. For AREA:
   - Focus on the main community/district name
   - Remove building names, street numbers, and unit details
   - If multiple areas are mentioned, choose the most specific one
   - For building complexes, use the community name
   - Handle compound names (e.g., "Mohammed Bin Rashid City")
   - Do not add numbers unless they're part of the official name

2. For EMIRATE:
   - Convert abbreviated forms (DXB → Dubai, AUH → Abu Dhabi, SHJ → Sharjah)
   - If emirate is not explicitly mentioned, infer from context if possible
   - Handle variations (Dubai/Dubayy/دبي → Dubai)

ANALYZE THIS ADDRESS: {location}

Return ONLY a JSON with two keys - 'area' and 'emirates'. Standardize to UPPERCASE. If unsure about any value, return an empty string for that field."""

    messages = [{"content": user_message, "role": "user"}]
    litellm.enable_json_schema_validation = True
    litellm.set_verbose = False
    
    try:
        resp = await litellm.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=messages,
            response_format=AddressEvent
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error processing address: {str(e)}")
        return {"area": "", "emirates": ""}

async def process_addresses(input_df: pd.DataFrame, master_df: pd.DataFrame):
    """Process each row from the input CSV as a separate address."""
    results_data = []
    
    for index, row in input_df.iterrows():
        try:
            # Extract metadata and delivery address
            po_number = row['PO Number']
            vendor = row['Vendor']
            delivery_address = row['delivery_address']
            comment = row['Comment']
            
            # Get LLM result for the delivery address
            llm_result = await test_get_response(delivery_address)
            logger.info(f"\nProcessing PO: {po_number}")
            logger.info(f"Input Address: {delivery_address}")
            logger.info(f"LLM Result: {llm_result}")
            
            # Match with master data
            matched_result = match_with_master(llm_result, delivery_address, master_df)
            logger.info(f"Matched Result: {matched_result}\n")
            
            # Prepare result row for the processed address
            result_row = {
                'PO_Number': po_number,
                'Vendor': vendor,
                'Input_Address': delivery_address,
                'Original_Comment': comment,
                'LLM_Area': llm_result.get('area', ''),
                'LLM_Emirate': llm_result.get('emirates', ''),
                'Matched_Area': matched_result['area'],
                'Matched_Emirate': matched_result.get('emirates', ''),
                'Matched_Zone': matched_result.get('zone', ''),
                'Matched_Row': matched_result.get('row_index', ''),  # Include the row index
                'Similarity_Score': matched_result.get('similarity_score', 0),
                'TEMP_Reason': matched_result.get('temp_reason', '')
            }
            results_data.append(result_row)
        
        except Exception as e:
            logger.error(f"Error processing row {index}: {str(e)}")
            results_data.append({
                'PO_Number': row.get('PO Number', ''),
                'Vendor': row.get('Vendor', ''),
                'Input_Address': row.get('delivery_address', ''),
                'Original_Comment': row.get('Comment', ''),
                'LLM_Area': '',
                'LLM_Emirate': '',
                'Matched_Area': 'TEMP',
                'Matched_Emirate': '',
                'Matched_Zone': '',
                'Matched_Row': '',
                'Similarity_Score': 0,
                'TEMP_Reason': f"Error processing address: {str(e)}"
            })
    
    # Create DataFrame with explicitly defined column order
    output_df = pd.DataFrame(results_data, columns=[
        'PO_Number', 'Vendor', 'Input_Address', 'Original_Comment', 
        'LLM_Area', 'LLM_Emirate', 'Matched_Area', 
        'Matched_Emirate', 'Matched_Zone', 'Matched_Row', 'Similarity_Score', 'TEMP_Reason'
    ])
    return output_df


def save_results_to_csv(results_df: pd.DataFrame) -> None:
    """Save results to CSV file with TEMP reasons"""
    try:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'Address_Matching_Results_{timestamp}.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        # Generate summary of TEMP reasons
        temp_reasons = results_df[results_df['Matched_Area'] == 'TEMP']['TEMP_Reason'].value_counts()
        summary_path = f'TEMP_Reasons_Summary_{timestamp}.txt'
        
        with open(summary_path, 'w') as f:
            f.write("Summary of TEMP Classifications:\n")
            f.write("=" * 50 + "\n\n")
            for reason, count in temp_reasons.items():
                f.write(f"{reason}: {count} occurrences\n")
            f.write(f"\nTotal TEMP addresses: {len(results_df[results_df['Matched_Area'] == 'TEMP'])}")
            f.write(f"\nTotal addresses processed: {len(results_df)}")
        
        logger.info(f"TEMP reasons summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")
        raise

async def main():
    """Main function to run the address matching process"""
    try:
        # Load the Area_master.csv
        master_csv_path = os.path.join(os.path.dirname(__file__), 'Area_master.csv')
        area_master_df = pd.read_csv(master_csv_path)
        logger.info(f"Successfully loaded Area_master.csv with {len(area_master_df)} rows")
        
        # Load input addresses
        input_csv_path = os.path.join(os.path.dirname(__file__), 'Address_input.csv')
        input_df = pd.read_csv(input_csv_path)
        logger.info(f"Successfully loaded {len(input_df)} addresses from Address_input.csv")
        
        # Process addresses and get results DataFrame
        results_df = await process_addresses(input_df, area_master_df)
        
        # Save results to CSV
        save_results_to_csv(results_df)
        
        # Print detailed summary
        print("\nProcessing Summary:")
        print("=" * 80)
        total_addresses = len(results_df)
        temp_addresses = len(results_df[results_df['Matched_Area'] == 'TEMP'])
        success_addresses = total_addresses - temp_addresses
        
        print(f"Total addresses processed: {total_addresses}")
        print(f"Successfully matched: {success_addresses}")
        print(f"Marked as TEMP: {temp_addresses}")
        print(f"Success rate: {(success_addresses/total_addresses)*100:.2f}%")
        
        print("\nTEMP Reasons Summary:")
        print("-" * 80)
        temp_reasons = results_df[results_df['Matched_Area'] == 'TEMP']['TEMP_Reason'].value_counts()
        for reason, count in temp_reasons.items():
            print(f"{reason}: {count} occurrences")
            
        # Additional Matching Details
        print("\nMatching Details:")
        print("-" * 80)
        exact_matches = len(results_df[results_df['Similarity_Score'] == 1.0])
        high_conf_matches = len(results_df[results_df['Similarity_Score'] >= 0.9])
        print(f"Exact matches: {exact_matches}")
        print(f"High confidence matches (≥0.9): {high_conf_matches}")
        
    except Exception as e:
        logger.error(f"Program error: {str(e)}")
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program terminated due to error: {str(e)}")
        logger.error(f"Program terminated due to error: {str(e)}")
