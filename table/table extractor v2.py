import fitz
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
import re
import json

def clean_table_text(text: str) -> str:
    """
    Clean up table text to ensure consistent formatting.
    """
    # Fix common OCR errors in "Table"
    text = re.sub(r'Tab[ilc]e', 'Table', text, flags=re.IGNORECASE)
    
    # Fix spaces between numbers and dots
    text = re.sub(r'(\d+)\s+(\d+)', r'\1.\2', text)
    text = re.sub(r'\.\s+\.', '.', text)
    
    # Fix table number format
    text = re.sub(r'Table\s+(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)', r'Table \1.\2.\3.\4', text)
    
    return text

def extract_text_with_robust_column_detection(pdf_path: str) -> str:
    """
    Robust column detection using k-means clustering and improved header/footer detection
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_dict = page.get_text("dict")
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Extract all text spans with positioning
        spans = []
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            # Clean up the text using the new function
                            text = clean_table_text(span["text"])
                            spans.append({
                                'text': text,
                                'x0': span["bbox"][0],
                                'y0': span["bbox"][1], 
                                'x1': span["bbox"][2],
                                'y1': span["bbox"][3],
                                'width': span["bbox"][2] - span["bbox"][0],
                                'height': span["bbox"][3] - span["bbox"][1]
                            })
        if not spans:
            continue
        # Step 1: Header detection (topmost, wide spans)
        header_spans = []
        body_spans = []
        # Heuristics: header = top 15% of page, wide
        header_y_thresh = page_height * 0.15
        for span in spans:
            if span['y0'] < header_y_thresh and span['width'] > page_width * 0.5:
                header_spans.append(span)
            else:
                body_spans.append(span)
        # Step 2: Column detection using k-means clustering on x0
        if not body_spans:
            columns = [[]]
        else:
            x0s = np.array([[span['x0']] for span in body_spans])
            if len(body_spans) > 10:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(x0s)
                labels = kmeans.labels_
                col0 = [span for span, label in zip(body_spans, labels) if label == 0]
                col1 = [span for span, label in zip(body_spans, labels) if label == 1]
                # Sort columns left-to-right by mean x0
                if np.mean([s['x0'] for s in col0]) < np.mean([s['x0'] for s in col1]):
                    left_col, right_col = col0, col1
                else:
                    left_col, right_col = col1, col0
                # Special handling for table names at column start
                def fix_column_start(column):
                    if not column:
                        return column
                    # Check first few spans for table names
                    for i, span in enumerate(column[:3]):
                        text = span['text'].strip()
                        if re.match(r'Tab[ilc]e\s+\d', text, re.IGNORECASE):
                            # Fix the text
                            span['text'] = re.sub(r'Tab[ilc]e', 'Table', text, flags=re.IGNORECASE)
                    return column
                
                left_col = fix_column_start(left_col)
                right_col = fix_column_start(right_col)
                
                # Sort each column by y0
                left_col.sort(key=lambda x: (x['y0'], x['x0']))
                right_col.sort(key=lambda x: (x['y0'], x['x0']))
                columns = [left_col, right_col]
            else:
                # Not enough spans, treat as single column
                columns = [sorted(body_spans, key=lambda x: (x['y0'], x['x0']))]
        # Step 3: Post-edit to remove header artifact from columns
        header_text = "ARTICLE 3.0 — GENERAL REQUIREMENTS FOR WIRING METHODS AND MATERIALS"
        def remove_header_artifact(column):
            return [span for span in column if header_text not in span['text'].replace('  ', ' ').replace('—', '—')]
        if len(columns) > 1:
            columns[0] = remove_header_artifact(columns[0])
            columns[1] = remove_header_artifact(columns[1])
        else:
            columns[0] = remove_header_artifact(columns[0])
        # Step 4: Remove trailing page number from columns
        def remove_trailing_page_number(column, page_num):
            if column and column[-1]['text'].strip() == str(page_num):
                return column[:-1]
            return column
        if len(columns) > 1:
            columns[0] = remove_trailing_page_number(columns[0], page_num + 1)
            columns[1] = remove_trailing_page_number(columns[1], page_num + 1)
        else:
            columns[0] = remove_trailing_page_number(columns[0], page_num + 1)
        # Step 5: Format output with only header, left, and right columns
        # Combine first line of left and right columns for header
        left_lines = group_spans_into_lines(columns[0]) if columns[0] else []
        right_lines = group_spans_into_lines(columns[1]) if len(columns) > 1 and columns[1] else []
        header_line = ""
        if left_lines and right_lines:
            header_line = "[HEADER]\n" + " ".join([span['text'] for span in left_lines[0]]) + " " + " ".join([span['text'] for span in right_lines[0]]) + "\n"
            left_lines = left_lines[1:]
            right_lines = right_lines[1:]
        elif left_lines:
            header_line = "[HEADER]\n" + " ".join([span['text'] for span in left_lines[0]]) + "\n"
            left_lines = left_lines[1:]
        elif right_lines:
            header_line = "[HEADER]\n" + " ".join([span['text'] for span in right_lines[0]]) + "\n"
            right_lines = right_lines[1:]
        # Format columns
        output = f"\n--- Page {page_num + 1} ---\n"
        output += header_line
        output += "[LEFT COLUMN]\n"
        for line in left_lines:
            line_text = " ".join([span['text'] for span in line])
            output += line_text + "\n"
        if len(columns) > 1:
            output += "[RIGHT COLUMN]\n"
            for line in right_lines:
                line_text = " ".join([span['text'] for span in line])
                output += line_text + "\n"
        # Normalize whitespace: replace multiple spaces with one, strip lines
        output = re.sub(r' +', ' ', output)
        output = '\n'.join([line.strip() for line in output.splitlines()])
        # Remove spaces before punctuation marks and after opening brackets
        output = re.sub(r'\s+([\.,;:!\?\)\]\}])', r'\1', output)  # before punctuation
        output = re.sub(r'([\(\[\{])\s+', r'\1', output)  # after opening bracket
        # Fix table/figure number patterns: remove spaces between digits, dots, and parentheses
        output = re.sub(
            r'(Table|Figure) ((?:\d+\.\s*)+\d+(?:\s*\([A-Za-z0-9]+\))*)',
            lambda m: m.group(1) + ' ' + re.sub(r'[\s]+', '', m.group(2)),
            output
        )
        # Additional cleaning for table numbers and names
        output = re.sub(r'Tab[ilc]e', 'Table', output, flags=re.IGNORECASE)  # Fix common OCR errors
        output = re.sub(r'(\d+)\s+(\d+)', r'\1.\2', output)  # Fix spaces between numbers
        output = re.sub(r'\.\s+\.', '.', output)  # Fix double dots with space
        # Clean the final output
        output = clean_table_text(output)
        full_text += output + "\n"
    doc.close()
    return full_text

def detect_columns_by_left_edges(spans: List[Dict], page_width: float) -> Dict:
    """
    Detect columns by analyzing the distribution of left edges (x0 values)
    More robust than center_x clustering
    """
    if not spans:
        return {'type': 'single', 'boundaries': []}
    
    # Extract left edges
    left_edges = [span['x0'] for span in spans]
    left_edges.sort()
    
    # Find significant gaps in left edge distribution
    gaps = []
    for i in range(1, len(left_edges)):
        gap = left_edges[i] - left_edges[i-1]
        gaps.append((gap, left_edges[i-1], left_edges[i]))
    
    # Find the largest gap that's significant (> 15% of page width)
    significant_gaps = [(gap, start, end) for gap, start, end in gaps 
                       if gap > page_width * 0.15]
    
    if not significant_gaps:
        return {'type': 'single', 'boundaries': []}
    
    # Take the largest significant gap as column separator
    largest_gap = max(significant_gaps, key=lambda x: x[0])
    separator_x = (largest_gap[1] + largest_gap[2]) / 2
    
    return {
        'type': 'double',
        'separator': separator_x,
        'boundaries': [0, separator_x, page_width]
    }

def assign_spans_to_columns(spans: List[Dict], column_info: Dict) -> List[List[Dict]]:
    """
    Assign spans to columns based on detected structure
    """
    if column_info['type'] == 'single':
        # Single column - sort by y position
        sorted_spans = sorted(spans, key=lambda x: (x['y0'], x['x0']))
        return [sorted_spans]
    
    # Double column
    separator = column_info['separator']
    left_column = []
    right_column = []
    
    for span in spans:
        if span['x0'] < separator:
            left_column.append(span)
        else:
            right_column.append(span)
    
    # Sort each column by vertical position
    left_column.sort(key=lambda x: (x['y0'], x['x0']))
    right_column.sort(key=lambda x: (x['y0'], x['x0']))
    
    return [left_column, right_column]

def clean_page_numbers(footer_spans: List[Dict], expected_page_num: int) -> List[Dict]:
    """
    Remove page numbers from footer spans
    """
    clean_spans = []
    for span in footer_spans:
        text = span['text'].strip()
        # Check if this looks like a page number
        if text.isdigit() and int(text) in [expected_page_num, expected_page_num - 1]:
            continue  # Skip page numbers
        clean_spans.append(span)
    return clean_spans

def format_page_output_grouped_no_footer(page_num: int, header_spans: List[Dict], 
                       columns: List[List[Dict]]) -> str:
    """
    Format the extracted text into readable output with only header, left, and right columns
    """
    output = f"\n--- Page {page_num} ---\n"
    # Add header if present
    if header_spans:
        output += "[HEADER]\n"
        for span in sorted(header_spans, key=lambda x: (x['y0'], x['x0'])):
            output += span['text'] + " "
        output += "\n"
    # Add left column
    if len(columns) > 1:
        output += "[LEFT COLUMN]\n"
        left_col = columns[0]
        lines = group_spans_into_lines(left_col)
        for line in lines:
            line_text = " ".join([span['text'] for span in line])
            output += line_text + "\n"
        # Add right column
        output += "[RIGHT COLUMN]\n"
        right_col = columns[1]
        lines = group_spans_into_lines(right_col)
        for line in lines:
            line_text = " ".join([span['text'] for span in line])
            output += line_text + "\n"
    else:
        # Single column fallback
        output += "[BODY]\n"
        lines = group_spans_into_lines(columns[0])
        for line in lines:
            line_text = " ".join([span['text'] for span in line])
            output += line_text + "\n"
    return output

def group_spans_into_lines(spans: List[Dict], line_threshold: float = 5.0) -> List[List[Dict]]:
    """
    Group spans that are vertically close into logical lines
    """
    if not spans:
        return []
    
    lines = []
    current_line = [spans[0]]
    
    for span in spans[1:]:
        # If this span is close vertically to the previous one, add to current line
        if abs(span['y0'] - current_line[-1]['y0']) <= line_threshold:
            current_line.append(span)
        else:
            # Start a new line
            lines.append(sorted(current_line, key=lambda x: x['x0']))  # Sort by x position
            current_line = [span]
    
    lines.append(sorted(current_line, key=lambda x: x['x0']))
    return lines

def find_table_references(text: str, table_list: List[str]) -> Dict[str, List[int]]:
    """
    Find table references in the text and map them to page numbers.
    Returns a dictionary mapping table numbers to lists of page numbers.
    """
    # Create a dictionary to store table numbers and their page numbers
    table_pages = defaultdict(list)
    
    # Split text into pages
    pages = text.split("--- Page ")
    pages = [p for p in pages if p.strip()]  # Remove empty pages
    
    for page in pages:
        # Extract page number
        page_num = int(page.split(" ---")[0])
        
        # Split page into lines
        lines = page.split("\n")
        
        # Look for table references at the start of lines
        for line in lines:
            line = line.strip()
            # Clean the line first
            line = clean_table_text(line)
            
            # Check if line starts with "Table"
            if line.startswith("Table "):
                # Extract the table number with improved pattern
                table_match = re.search(r'Table\s+([\d\.]+(?:\([A-Za-z0-9]+\))*)', line)
                if table_match:
                    table_num = table_match.group(1)
                    # Clean up the table number
                    table_num = re.sub(r'\s+', '', table_num)  # Remove any remaining spaces
                    # Check if this table number is in our list
                    if table_num in table_list:
                        table_pages[table_num].append(page_num)
    
    return dict(table_pages)

def load_table_list(file_path: str) -> List[str]:
    """
    Load the list of table numbers from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Remove the square brackets and split by commas
        content = f.read().strip('[]\n')
        # Split by comma and clean up each item
        table_list = [item.strip().strip('"') for item in content.split(',')]
        return table_list

# Example usage
if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = "PEC_Content_1-2.pdf"
    extracted_text = extract_text_with_robust_column_detection(pdf_path)
    
    # Save the extracted text with debug info
    with open("PEC_Content_1-2_debug.txt", "w", encoding="utf-8") as f:
        f.write("=== EXTRACTED TEXT WITH DEBUG INFO ===\n\n")
        f.write(extracted_text)
        f.write("\n\n=== TABLE REFERENCES FOUND ===\n\n")
        
        # Split text into pages for debugging
        pages = extracted_text.split("--- Page ")
        pages = [p for p in pages if p.strip()]  # Remove empty pages
        
        for page in pages:
            # Extract page number
            page_num = int(page.split(" ---")[0])
            f.write(f"\n--- Page {page_num} ---\n")
            
            # Look for table references
            lines = page.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("Table "):
                    f.write(f"Found: {line}\n")
    
    # Load the table list
    table_list = load_table_list("unique_tables_v3.txt")
    
    # Find table references and their page numbers
    table_pages = find_table_references(extracted_text, table_list)
    
    # Format output as single line per table
    output_lines = []
    for table_num, pages in sorted(table_pages.items()):
        # Remove duplicates and sort page numbers
        unique_pages = sorted(set(pages))
        # Format as "table_number: page1, page2, page3"
        output_lines.append(f"{table_num}: {', '.join(map(str, unique_pages))}")
    
    # Save the table-page mapping
    with open("table_pages_1-2.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))