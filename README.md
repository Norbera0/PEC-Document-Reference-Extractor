# PEC Document Reference Extractor

A Python tool for extracting and analyzing various types of references from Philippine Electrical Code (PEC) documents, including tables, figures, and other document elements.

## Features

- Extracts multiple types of references from PDF documents:
  - Table numbers (e.g., "Table 3.10.2.51(C)(71)")
  - Figure numbers
  - Section references
  - Other document elements
- Handles complex reference formats with:
  - Nested numbering (e.g., "1.10.2.1(A)")
  - Multiple levels (e.g., "2.20.4.5")
  - Special characters and parentheses
- Supports both simple and detailed extraction modes
- Generates debug information for analysis
- Saves results in various formats (JSON, TXT)
- Provides reference validation and verification

## Requirements

- Python 3.6+
- PyMuPDF (fitz)
- scikit-learn (for advanced column detection)
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pec-document-extractor.git
cd pec-document-extractor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from document_extractor import extract_unique_numbers

# Extract references from a PDF
pdf_path = "your_document.pdf"

# Extract table numbers
table_numbers = extract_unique_numbers(pdf_path, label="Table")

# Extract figure numbers
figure_numbers = extract_unique_numbers(pdf_path, label="Figure")

# Print results
print("Tables found:")
for num in table_numbers:
    print(f"Table {num}")

print("\nFigures found:")
for num in figure_numbers:
    print(f"Figure {num}")
```

### Debug Mode

```python
from document_extractor_debug import extract_text_with_debug

# Extract with detailed debugging information
debug_info = extract_text_with_debug(pdf_path)
```

## Project Structure

```
pec-document-extractor/
├── README.md
├── requirements.txt
├── document_extractor.py       # Main extraction module
├── document_extractor_debug.py # Debug version with detailed logging
└── hub/                        # Additional resources
    ├── unique_tables_v3.txt    # Reference table numbers
    ├── unique_figures_v3.txt   # Reference figure numbers
    └── reference_data/         # Additional reference data
```

## Output Format

The tool generates two types of output:

1. **Simple Mode**: A list of unique references by type
2. **Debug Mode**: Detailed information including:
   - Raw text from each page
   - Text spans with positioning
   - Matches found on each page
   - Final sorted results
   - Reference validation results

## Reference Types

The extractor handles various types of references:

1. **Tables**: Numbered references to tables in the document
2. **Figures**: Numbered references to figures and diagrams
3. **Sections**: References to document sections and subsections
4. **Special References**: References with special formatting or notation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyMuPDF for PDF processing capabilities
- scikit-learn for advanced text analysis features
- Philippine Electrical Code (PEC) for document structure reference 