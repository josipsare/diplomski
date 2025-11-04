# Quick Start Guide - SEC Financial Data

## Installation

1. **Install Python** (3.8 or higher)
   ```bash
   python --version
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Step 1: Download Data

```python
from sec_data_downloader import SECDataDownloader

# Your information
USER_AGENT = "Josip Sare josip.sare@gmail.com"

downloader = SECDataDownloader(
    user_agent=USER_AGENT,
    output_dir="./sec_data"
)

# Download latest quarter
downloader.download_latest()
```

### Step 2: Parse Data

```python
from sec_data_parser import SECDataParser

# Point to the downloaded quarter directory
parser = SECDataParser("./sec_data/2024q4")

# Get statistics
stats = parser.get_statistics()
print(stats)
```

### Step 3: Search for a Company

```python
# Search by name
companies = parser.get_company_by_name("Apple")
print(companies[['cik', 'name']])

# Get Apple's CIK
apple_cik = companies.iloc[0]['cik']
```

### Step 4: Get Financial Data

```python
# Get company submissions
submissions = parser.get_company_submissions(apple_cik)
print(submissions[['form', 'period', 'filed']])

# Get financial summary
summary = parser.get_financial_summary(apple_cik, form='10-K')
print(summary)
```

### Step 5: Export Data

```python
# Export to CSV
parser.export_to_csv(summary, "apple_financials.csv")
```

## Run Complete Example

```bash
python example_usage.py
```

## Common Tasks

### Download Specific Quarter
```python
downloader.download_quarter(2024, 4)  # Q4 2024
```

### Download Entire Year
```python
downloader.download_year(2024)
```

### Download Multiple Years
```python
downloader.download_range(2022, 2024)
```

### Search for XBRL Tags
```python
# Find revenue-related tags
revenue_tags = parser.search_tags("revenue")
print(revenue_tags)
```

### Get Specific Filing Data
```python
# Get data for a specific filing (accession number)
filing_data = parser.get_filing_data("0000320193-24-000123")
print(filing_data['numbers'])  # Numerical facts
print(filing_data['text'])     # Text facts
```

### Filter by Form Type
```python
# Get only 10-K (annual) filings
financials = parser.get_company_financials(
    cik=apple_cik,
    forms=['10-K']
)
```

### Filter by Specific Tags
```python
# Get only specific financial metrics
financials = parser.get_company_financials(
    cik=apple_cik,
    tags=['Assets', 'Revenues', 'NetIncomeLoss']
)
```

## Important Notes

1. **User-Agent is Required**: The SEC requires a User-Agent header with your email
   - Format: `"CompanyName contact@email.com"`
   - Requests without proper User-Agent will be blocked

2. **Rate Limiting**: Respect SEC rate limits (10 requests/second max)
   - The downloader automatically handles this

3. **Data Size**: Each quarter is 200-400 MB compressed, 2-4 GB uncompressed
   - Plan disk space accordingly

4. **File Format**: Files are tab-delimited (.txt), not comma-delimited
   - The parser handles this automatically

5. **CIK Format**: CIKs can be with or without leading zeros
   - "0000320193" or "320193" both work

## Common Company CIKs

- Apple Inc: 0000320193
- Microsoft: 0000789019
- Tesla: 0001318605
- Amazon: 0001018724
- Google/Alphabet: 0001652044
- Meta/Facebook: 0001326801
- NVIDIA: 0001045810

## Troubleshooting

### 403 Error
- Make sure User-Agent header is set correctly with email
- Check that you're not exceeding rate limits

### File Not Found
- Ensure the quarter exists (data starts from 2009)
- Some quarters may not be available yet

### Memory Issues
- Large files require significant RAM
- Process data in chunks if needed
- Use filters to reduce data size

## Next Steps

1. Read the full documentation: [README_SEC_DATA.md](README_SEC_DATA.md)
2. Explore example scripts: [example_usage.py](example_usage.py)
3. Check SEC's official documentation: https://www.sec.gov/files/aqfs.pdf

## Support

- SEC Data: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
- SEC API Docs: https://www.sec.gov/edgar/sec-api-documentation
- GitHub Issues: Report issues on your project's GitHub page
