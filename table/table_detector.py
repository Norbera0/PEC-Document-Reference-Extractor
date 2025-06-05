import fitz  # PyMuPDF
import json
from collections import defaultdict
import re

def is_likely_table(text_blocks, min_blocks=2):
    """
    Analyze text blocks to determine if they form a table-like structure.
    Returns True if the blocks show table-like characteristics.
    """
    if len(text_blocks) < min_blocks:
        return False
    
    # Get x-coordinates of blocks
    x_coords = [block[0] for block in text_blocks]  # x0 coordinates
    x_coords.sort()
    
    # Check for column alignment (similar x-coordinates)
    x_groups = []
    current_group = [x_coords[0]]
    
    for x in x_coords[1:]:
        if abs(x - current_group[-1]) < 20:  # Reduced tolerance
            current_group.append(x)
        else:
            if len(current_group) >= 2:  # At least 2 blocks aligned
                x_groups.append(current_group)
            current_group = [x]
    
    if len(current_group) >= 2:
        x_groups.append(current_group)
    
    # Check for row structure
    y_coords = [block[1] for block in text_blocks]  # y0 coordinates
    y_coords.sort()
    
    y_groups = []
    current_group = [y_coords[0]]
    
    for y in y_coords[1:]:
        if abs(y - current_group[-1]) < 25:  # Reduced tolerance
            current_group.append(y)
        else:
            if len(current_group) >= 2:  # At least 2 blocks in a row
                y_groups.append(current_group)
            current_group = [y]
    
    if len(current_group) >= 2:
        y_groups.append(current_group)
    
    # Check for table-like characteristics
    has_columns = len(x_groups) >= 2  # At least 2 columns
    has_rows = len(y_groups) >= 2     # At least 2 rows
    
    # Check for uniform spacing
    if has_columns and has_rows:
        # Calculate average spacing between rows
        row_spacings = []
        for group in y_groups:
            if len(group) > 1:
                spacing = sum(abs(group[i] - group[i-1]) for i in range(1, len(group))) / (len(group) - 1)
                row_spacings.append(spacing)
        
        # Check if spacings are relatively uniform
        if row_spacings:
            avg_spacing = sum(row_spacings) / len(row_spacings)
            uniform_spacing = all(abs(s - avg_spacing) < avg_spacing * 0.4 for s in row_spacings)
        else:
            uniform_spacing = False
            
        return has_columns and has_rows and uniform_spacing
    
    return False

def analyze_pages_for_tables(pdf_path, table_pages):
    """
    Analyze specified pages for actual table structures.
    Returns a dictionary mapping table numbers to their detected pages.
    """
    doc = fitz.open(pdf_path)
    table_detections = {table: [] for table in table_pages.keys()}
    
    # Analyze each page
    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]  # Convert to 0-based index
        
        # Get text blocks with their coordinates
        blocks = page.get_text("blocks")
        
        # Filter blocks to get only those that might be part of a table
        table_blocks = []
        for block in blocks:
            text = block[4].strip()
            # More strict text length filtering
            if len(text) < 2 or len(text) > 500:
                continue
            # More strict header/footer filtering
            if block[1] < 50 or block[3] > page.rect.height - 50:
                continue
            table_blocks.append(block)
        
        # Check if the blocks form a table-like structure
        if is_likely_table(table_blocks):
            # Find which table(s) this page belongs to
            for table, pages in table_pages.items():
                if page_num in pages:
                    table_detections[table].append(page_num)
    
    doc.close()
    return table_detections

def create_table_pages_pdf(pdf_path, pages_with_tables, output_pdf_path):
    """
    Create a new PDF containing only the pages with actual tables.
    """
    doc = fitz.open(pdf_path)
    output = fitz.open()
    
    for page_num in pages_with_tables:
        output.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
    
    output.save(output_pdf_path)
    output.close()
    doc.close()

def main():
    # Load the table pages from the previous output
    with open("PEC_Content_1-4_tables.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse the table pages
    table_pages = {}
    for line in content.split('\n'):
        if ':' in line:
            table, pages = line.split(':')
            table = table.strip()
            # Remove 'pages' prefix and split by comma
            pages = pages.replace('pages', '').strip()
            if pages:
                page_list = [int(p.strip()) for p in pages.split(',') if p.strip().isdigit()]
                table_pages[table] = page_list
    
    # Analyze pages for actual tables
    pdf_path = "PEC_Content_1-4.pdf"
    table_detections = analyze_pages_for_tables(pdf_path, table_pages)
    
    # Create output PDF with table pages
    all_detected_pages = set()
    for pages in table_detections.values():
        all_detected_pages.update(pages)
    
    output_pdf_path = "PEC_Content_tables_v5.pdf"
    create_table_pages_pdf(pdf_path, sorted(all_detected_pages), output_pdf_path)
    
    # Save the list of tables and their detected pages
    with open("PEC_Content_1-4_tables_v5.txt", "w", encoding="utf-8") as f:
        for table in sorted(table_pages.keys()):
            pages = table_detections[table]
            if pages:
                pages_str = ", ".join(map(str, sorted(pages)))
                f.write(f"{table}: pages {pages_str}\n")
            else:
                f.write(f"{table}: NA\n")
    
    print(f"Found {len(all_detected_pages)} pages with actual tables")
    print(f"Output saved to 'PEC_Content_1-4_tables_v3.txt'")
    print(f"PDF with table pages saved to '{output_pdf_path}'")

if __name__ == "__main__":
    main() 