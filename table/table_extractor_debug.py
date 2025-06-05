import fitz  # PyMuPDF
import re
import json
import os
from typing import List, Dict, Set
from datetime import datetime

def log_debug(message: str, data: any = None):
    """Helper function to log debug information"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[{timestamp}] {message}")
    if data is not None:
        print(f"Data: {data}")

def extract_text_with_debug(pdf_path: str, label: str = "Table") -> Dict[str, List[str]]:
    """
    Extract text with detailed debugging information at each step.
    Returns a dictionary with debug information and results.
    """
    debug_info = {
        "raw_text": [],
        "spans": [],
        "matches": [],
        "final_results": []
    }
    
    doc = fitz.open(pdf_path)
    pattern = re.compile(rf'\b{label}\s+(\d[^\s]*(?:\d|\)|-\d+))', re.IGNORECASE)
    unique_numbers = set()

    for page_num in range(len(doc)):
        log_debug(f"Processing page {page_num + 1}")
        
        # Get raw text
        text = doc[page_num].get_text("text")
        debug_info["raw_text"].append({
            "page": page_num + 1,
            "text": text
        })
        log_debug(f"Raw text from page {page_num + 1}", text[:200] + "..." if len(text) > 200 else text)
        
        # Get text with position information
        text_dict = doc[page_num].get_text("dict")
        spans = []
        
        # Extract spans with position
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            spans.append({
                                "text": span["text"],
                                "bbox": span["bbox"],
                                "font": span.get("font", "unknown"),
                                "size": span.get("size", 0)
                            })
        
        debug_info["spans"].append({
            "page": page_num + 1,
            "spans": spans
        })
        log_debug(f"Found {len(spans)} text spans on page {page_num + 1}")
        
        # Find matches
        matches = pattern.findall(text)
        debug_info["matches"].append({
            "page": page_num + 1,
            "matches": matches
        })
        log_debug(f"Found {len(matches)} matches on page {page_num + 1}", matches)
        
        # Store unique numbers
        for match in matches:
            unique_numbers.add(match.strip())
    
    doc.close()
    
    # Sort the results
    def sort_key(x):
        parts = []
        for part in x.split('.'):
            components = re.findall(r'(\d+|[^\d]+)', part)
            parts.append(tuple(components))
        return parts
    
    sorted_list = sorted(unique_numbers, key=sort_key)
    debug_info["final_results"] = sorted_list
    
    return debug_info

def save_debug_info(debug_info: Dict, output_dir: str):
    """Save debug information to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create debug directory if it doesn't exist
    debug_dir = os.path.join(output_dir, f"debug_{timestamp}")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save raw text
    with open(os.path.join(debug_dir, "raw_text.txt"), "w", encoding="utf-8") as f:
        for page_info in debug_info["raw_text"]:
            f.write(f"\n=== Page {page_info['page']} ===\n")
            f.write(page_info["text"])
    
    # Save spans
    with open(os.path.join(debug_dir, "spans.txt"), "w", encoding="utf-8") as f:
        for page_info in debug_info["spans"]:
            f.write(f"\n=== Page {page_info['page']} Spans ===\n")
            for span in page_info["spans"]:
                f.write(f"Text: '{span['text']}'\n")
                f.write(f"Position: {span['bbox']}\n")
                f.write(f"Font: {span['font']}, Size: {span['size']}\n")
                f.write("-" * 50 + "\n")
    
    # Save matches
    with open(os.path.join(debug_dir, "matches.txt"), "w", encoding="utf-8") as f:
        for page_info in debug_info["matches"]:
            f.write(f"\n=== Page {page_info['page']} Matches ===\n")
            for match in page_info["matches"]:
                f.write(f"{match}\n")
    
    # Save final results
    with open(os.path.join(debug_dir, "final_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n=== Final Results ===\n")
        for result in debug_info["final_results"]:
            f.write(f"{result}\n")
    
    return debug_dir

def main():
    # Replace with your PDF path
    pdf_path = "PEC_Content_1-2.pdf"
    output_dir = os.path.dirname(pdf_path)
    
    print("Starting debug extraction...")
    debug_info = extract_text_with_debug(pdf_path)
    
    print("\nSaving debug information...")
    debug_dir = save_debug_info(debug_info, output_dir)
    
    print(f"\nDebug information saved to: {debug_dir}")
    print(f"Total unique numbers found: {len(debug_info['final_results'])}")
    
    # Print final results
    print("\nFinal Results:")
    for result in debug_info["final_results"]:
        print(f"Table {result}")

if __name__ == "__main__":
    main() 