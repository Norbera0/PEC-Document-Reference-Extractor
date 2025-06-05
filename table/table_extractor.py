import fitz  # PyMuPDF
import json
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans

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
                            spans.append({
                                'text': span["text"],
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
        # Remove trailing page number from columns (again, after all cleaning)
        def remove_trailing_page_number_from_lines(lines, page_num):
            if lines and len(lines[-1]) == 1 and lines[-1][0]['text'].strip() == str(page_num):
                return lines[:-1]
            return lines
        left_lines = remove_trailing_page_number_from_lines(left_lines, page_num + 1)
        if len(columns) > 1:
            right_lines = remove_trailing_page_number_from_lines(right_lines, page_num + 1)
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
        full_text += output + "\n"
    doc.close()
    return full_text

# === Load table numbers from file ===
with open("unique_tables_v3.txt", "r", encoding="utf-8") as f:
    table_numbers = json.load(f)

# Sort table numbers by length (longest first) to ensure we match the most specific number first
table_numbers.sort(key=len, reverse=True)
table_numbers_set = set(table_numbers)

# === File paths ===
pdf_path = "PEC_TTest.pdf"
output_txt_path = "PEC_TTest_tables.txt"

# Extract text with robust column detection
extracted_text = extract_text_with_robust_column_detection(pdf_path)

# Save the extracted text first
with open("PEC_TTest.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

# Process the extracted text to find tables
tables_found = {}
lines = extracted_text.split('\n')

for line in lines:
    line = line.strip()
    if line.lower().startswith('table'):
        for table in table_numbers_set:
            pattern = rf'^Table\s+{re.escape(table)}(\b|\s|$)'
            if re.search(pattern, line, re.IGNORECASE):
                if table not in tables_found:
                    tables_found[table] = []
                # Extract page number from the line above
                for i in range(len(lines)):
                    if lines[i] == line:
                        if i > 0 and "Page" in lines[i-1]:
                            page_match = re.search(r'Page (\d+)', lines[i-1])
                            if page_match:
                                page_num = int(page_match.group(1))
                                tables_found[table].append(page_num)
                        break

# Save results
with open(output_txt_path, "w", encoding="utf-8") as f:
    for table, pages in sorted(tables_found.items()):
        pages_str = ", ".join(map(str, sorted(set(pages))))
        f.write(f"{table}: pages {pages_str}\n")

print(f"Table page list saved to: {output_txt_path}")
print(f"Total unique tables found: {len(tables_found)}")
