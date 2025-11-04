# SEC Financial Statement Data Sets - Download & Parse Guide

## Overview

The SEC provides quarterly financial statement data sets extracted from XBRL filings. This data includes numerical facts, text narratives, and metadata from all public company filings.

## Data Structure

### Quarterly ZIP Files
- **URL Pattern**: `https://www.sec.gov/files/dera/data/financial-statement-data-sets/{YEAR}q{QUARTER}.zip`
- **Examples**:
  - Q1 2025: `2025q1.zip`
  - Q4 2024: `2024q4.zip`
  - Q3 2023: `2023q3.zip`

### CSV Files in Each ZIP

Each quarterly ZIP contains 4 tab-delimited text files:

1. **SUB.txt** - Submissions
   - Company information (CIK, company name, SIC, IRS number)
   - Filing details (form type, filing date, period)
   - ~100,000+ submissions per quarter

2. **NUM.txt** - Numerical Facts
   - Financial metrics (revenue, assets, liabilities, etc.)
   - Contains: tag, value, units, decimals, context
   - Millions of rows per quarter

3. **TAG.txt** - Tag Definitions
   - Taxonomy information for each XBRL tag
   - Tag descriptions and metadata
   - ~15,000+ unique tags

4. **TXT.txt** - Text Facts
   - Narrative disclosures and text-based facts
   - Accounting policies, risk factors, etc.

## Key Fields

### SUB.txt (Submissions)
- `adsh` - Accession number (unique filing ID)
- `cik` - Central Index Key (company identifier)
- `name` - Company name
- `sic` - Standard Industrial Classification
- `form` - Filing type (10-K, 10-Q, 8-K, etc.)
- `period` - Reporting period end date
- `filed` - Filing date

### NUM.txt (Numerical Facts)
- `adsh` - Accession number (links to SUB)
- `tag` - XBRL tag name (e.g., "Assets", "Revenue")
- `value` - Numerical value
- `uom` - Unit of measure (USD, shares, etc.)
- `ddate` - Data date/period

### TAG.txt (Tag Definitions)
- `tag` - XBRL tag name
- `version` - Taxonomy version
- `tlabel` - Tag label/description
- `datatype` - Data type

### TXT.txt (Text Facts)
- `adsh` - Accession number
- `tag` - XBRL tag name
- `value` - Text content

## Access Requirements

### User-Agent Header (Required)
Per SEC fair access policy, you MUST include:
```
User-Agent: YourCompanyName AdminContact@example.com
```

### Rate Limits
- Maximum: 10 requests per second
- Recommended: 1-2 requests per second for safety

## Usage

### Basic Download
```python
from sec_data_downloader import SECDataDownloader

downloader = SECDataDownloader(
    user_agent="MyCompany admin@example.com",
    output_dir="./sec_data"
)

# Download single quarter
downloader.download_quarter(2024, 4)

# Download multiple quarters
downloader.download_year(2024)
```

### Parse Data
```python
from sec_data_parser import SECDataParser

parser = SECDataParser("./sec_data/2024q4")

# Load submissions
submissions = parser.load_submissions()

# Get specific company data
apple_data = parser.get_company_data(cik="0000320193")

# Get financials for a company
financials = parser.get_company_financials(cik="0000320193")
```

## Data Size

- Single quarter ZIP: ~200-400 MB compressed, ~2-4 GB uncompressed
- Annual data (4 quarters): ~1-2 GB compressed, ~8-16 GB uncompressed

## Resources

- Official SEC Page: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
- SEC Python Examples: https://github.com/sec-gov/python-for-dera-financial-datasets
- Data Dictionary: https://www.sec.gov/files/aqfs.pdf

## Notes

- Data available from 2009 onwards
- Files are updated monthly
- Tab-delimited format (not comma-delimited)
- Large files require sufficient memory for processing
