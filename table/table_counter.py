import fitz  # PyMuPDF
import re
import json
import os

def extract_unique_numbers(pdf_path, label="Figure"):
    doc = fitz.open(pdf_path)
    # Pattern that captures anything after "Table " that starts with a number
    # and ends with either a number, parenthesis, or hyphen-number
    pattern = re.compile(rf'\b{label}\s+(\d[^\s]*(?:\d|\)|-\d+))', re.IGNORECASE)

    unique_numbers = set()

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        matches = pattern.findall(text)
        for match in matches:
            # Store the complete sequence as is
            unique_numbers.add(match.strip())

    doc.close()

    # Custom sorting function to handle mixed alphanumeric sequences
    def sort_key(x):
        # Split by dots and convert each part to a tuple of (number, string)
        parts = []
        for part in x.split('.'):
            # Split each part into number and non-number components
            components = re.findall(r'(\d+|[^\d]+)', part)
            parts.append(tuple(components))
        return parts

    sorted_list = sorted(unique_numbers, key=sort_key)
    return sorted_list

def save_list_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    pdf_path = "PEC_Content_1-4.pdf"  # Replace with your path

    # Extract
    unique_figures = extract_unique_numbers(pdf_path, label="Figure")
    unique_tables = extract_unique_numbers(pdf_path, label="Table")

    # Print to console
    print("📋 Unique Figure Numbers Found:")
    for num in unique_figures:
        print(f"Figure {num}")
    print(f"\n🔢 Total Unique Figures: {len(unique_figures)}\n")

    print("📋 Unique Table Numbers Found:")
    for num in unique_tables:
        print(f"Table {num}")
    print(f"\n🔢 Total Unique Tables: {len(unique_tables)}")

    # Save to files
    base_dir = os.path.dirname(pdf_path)
    save_list_to_file(unique_figures, os.path.join(base_dir, "unique_figures_v3.txt"))
    save_list_to_file(unique_tables, os.path.join(base_dir, "unique_tables_v3.txt"))
    print("\n💾 Output saved to 'unique_figures_v2.txt' and 'unique_tables_v3.txt'.")

if __name__ == "__main__":
    main()
