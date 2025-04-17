from plistlib import InvalidFileException
import requests
import pandas as pd
from striprtf.striprtf import rtf_to_text
import re
import time
import os
from openpyxl import load_workbook
import pandas as pd
import json


def load_excel_data(excel_file_path):
    # Load data from the Excel file
    df = pd.read_excel(excel_file_path)
    df.columns = df.columns.str.strip()
    df['Post ID'] = df['URL'].str.extract('(\d+)$')  # Extract Post IDs from URLs
    return df

def extract_post_ids_from_rtf(rtf_file_path):
    # Read the RTF file and extract text
    with open(rtf_file_path, 'r') as file:
        rtf_content = file.read()
    text_content = rtf_to_text(rtf_content)
    # Extract Post IDs using regular expressions
    return re.findall(r'\b\d+\b', text_content)

def fetch_data_from_stackoverflow(post_ids):
    data = []
    for post_id in post_ids:
        url = f"https://api.stackexchange.com/2.3/posts/{post_id}"
        params = {
            'order': 'desc',
            'sort': 'activity',
            'site': 'stackoverflow',
            'filter': 'withbody'  # Replace 'withbody' with your custom filter ID
        }
        print("print url",url)
        result = make_request(url,params=params)
       
        if result!=None: 
            if 'items' in result and len(result['items']) > 0:
                print("result",result)
                item = result['items'][0]
                data.append({
                    "Post ID": post_id,
                    "Title": item.get('title', 'No title available'),
                    "Question": item.get('body', 'No question content available'),
                    "Highest Voted Answer": item['answers'][0]['body'] if 'answers' in item and len(item['answers']) > 0 else 'No answers available'
                })
                
    return pd.DataFrame(data)

def make_request(url, params):
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("success")
            time.sleep(1)
            return response.json()    
        else:
            print("failed")
            time.sleep(5) 
             # Wait for 5 seconds before retrying
            make_request(url, params)  # Recursive retry
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def combine_data(excel_data, rtf_ids, stackoverflow_data):
    # Mark all RTF IDs with empty Purpose and Solution
    rtf_df = pd.DataFrame({
        "Post ID": rtf_ids,
        "Purpose": '',
        "Solution": ''
    })
    # Combine Excel data with RTF data
    combined_df = pd.concat([excel_data[['Post ID', 'Purpose', 'Solution']], rtf_df], ignore_index=True)
    # Merge with Stack Overflow data
    final_df = pd.merge(combined_df, stackoverflow_data, on="Post ID", how="left")
    return final_df

def save_to_excel(data, excel_file_path_1):
    print("Saving to Excel")
    excel_file_path=excel_file_path_1+"preprocessod_data.xlsx"
    temp_excel_file =excel_file_path_1+"preprocessod_data_backup.xlsx"
    if os.path.exists(excel_file_path):
        data.to_excel(temp_excel_file, index=False)
    else:
        print("File not found, creating a new file.")
        data.to_excel(excel_file_path, index=False)

      

excel_file_path = './assets/Architectural_Posts.xlsx'
rtf_file_path = './assets/Programming_posts.rtf'
output_excel_file_path = './datasets/'


# Load and process data
excel_data = load_excel_data(excel_file_path)
rtf_ids = extract_post_ids_from_rtf(rtf_file_path)
#post_ids = list(set(excel_data['Post ID'].tolist() + rtf_ids)) 
post_ids = list(set(rtf_ids))  # Combine and remove duplicates
stackoverflow_data = fetch_data_from_stackoverflow([post_ids[1]])
final_data = combine_data(excel_data, rtf_ids, stackoverflow_data)

# Save the final data
#manage_json_data(json_data,final_data)
save_to_excel(final_data, output_excel_file_path)
