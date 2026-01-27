"""
SEC Financial Statement Data Parser

Parse and analyze downloaded SEC financial statement data sets.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
import warnings


class SECDataParser:
    """
    Parse SEC financial statement data sets.

    Loads and provides convenient access to SUB, NUM, TAG, and TXT files.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the parser.

        Args:
            data_dir: Directory containing extracted SEC data files
                     (should contain SUB.txt, NUM.txt, TAG.txt, TXT.txt)
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # File paths
        self.sub_file = self.data_dir / "sub.txt"
        self.num_file = self.data_dir / "num.txt"
        self.tag_file = self.data_dir / "tag.txt"
        self.txt_file = self.data_dir / "txt.txt"

        # Cached dataframes
        self._sub_df: Optional[pd.DataFrame] = None
        self._num_df: Optional[pd.DataFrame] = None
        self._tag_df: Optional[pd.DataFrame] = None
        self._txt_df: Optional[pd.DataFrame] = None

    def _read_tsv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Read tab-separated file.

        Args:
            file_path: Path to file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Loading: {file_path.name}...", end=" ")

        # Default settings for SEC files
        default_kwargs = {
            'sep': '\t',
            'dtype': str,  # Load as strings initially to avoid type issues
            'encoding': 'utf-8',
            'low_memory': False
        }
        default_kwargs.update(kwargs)

        df = pd.read_csv(file_path, **default_kwargs)
        print(f"Done ({len(df):,} rows)")

        return df

    def load_submissions(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load submissions (SUB) data.

        Contains company and filing information.

        Key columns:
            - adsh: Accession number (unique filing ID)
            - cik: Central Index Key (company ID)
            - name: Company name
            - sic: Standard Industrial Classification code
            - form: Filing type (10-K, 10-Q, etc.)
            - period: Reporting period end date
            - filed: Filing date

        Args:
            force_reload: Force reload from file

        Returns:
            DataFrame with submission data
        """
        if self._sub_df is None or force_reload:
            self._sub_df = self._read_tsv(self.sub_file)

            # Convert date columns
            date_cols = ['period', 'filed', 'accepted']
            for col in date_cols:
                if col in self._sub_df.columns:
                    self._sub_df[col] = pd.to_datetime(self._sub_df[col], errors='coerce')

            # Convert numeric columns
            if 'cik' in self._sub_df.columns:
                self._sub_df['cik'] = pd.to_numeric(self._sub_df['cik'], errors='coerce')

        return self._sub_df

    def load_numbers(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load numerical facts (NUM) data.

        Contains financial metrics and values.

        Key columns:
            - adsh: Accession number (links to submissions)
            - tag: XBRL tag name
            - value: Numerical value
            - uom: Unit of measure (USD, shares, etc.)
            - ddate: Data date

        Args:
            force_reload: Force reload from file

        Returns:
            DataFrame with numerical data
        """
        if self._num_df is None or force_reload:
            self._num_df = self._read_tsv(self.num_file)

            # Convert value to float
            if 'value' in self._num_df.columns:
                self._num_df['value'] = pd.to_numeric(self._num_df['value'], errors='coerce')

            # Convert date
            if 'ddate' in self._num_df.columns:
                self._num_df['ddate'] = pd.to_datetime(self._num_df['ddate'], errors='coerce')

        return self._num_df

    def load_tags(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load tag definitions (TAG) data.

        Contains XBRL tag metadata and descriptions.

        Key columns:
            - tag: XBRL tag name
            - version: Taxonomy version
            - tlabel: Tag label/description
            - datatype: Data type

        Args:
            force_reload: Force reload from file

        Returns:
            DataFrame with tag data
        """
        if self._tag_df is None or force_reload:
            self._tag_df = self._read_tsv(self.tag_file)

        return self._tag_df

    def load_text(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load text facts (TXT) data.

        Contains narrative disclosures and text-based facts.

        Key columns:
            - adsh: Accession number
            - tag: XBRL tag name
            - value: Text content

        Args:
            force_reload: Force reload from file

        Returns:
            DataFrame with text data
        """
        if self._txt_df is None or force_reload:
            self._txt_df = self._read_tsv(self.txt_file)

        return self._txt_df

    def get_company_submissions(self, cik: Union[str, int]) -> pd.DataFrame:
        """
        Get all submissions for a specific company.

        Args:
            cik: Central Index Key (with or without leading zeros)

        Returns:
            DataFrame with company's submissions
        """
        sub_df = self.load_submissions()

        # Convert CIK to integer for comparison
        cik_int = int(str(cik).lstrip('0')) if cik else 0

        return sub_df[sub_df['cik'] == cik_int].copy()

    def get_company_by_name(self, name: str, exact: bool = False) -> pd.DataFrame:
        """
        Search for companies by name.

        Args:
            name: Company name or partial name
            exact: Whether to match exactly (case-insensitive)

        Returns:
            DataFrame with matching companies
        """
        sub_df = self.load_submissions()

        if exact:
            mask = sub_df['name'].str.lower() == name.lower()
        else:
            mask = sub_df['name'].str.contains(name, case=False, na=False)

        return sub_df[mask][['cik', 'name', 'sic', 'form', 'period', 'filed']].drop_duplicates('cik')

    def get_filing_data(self, adsh: str) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific filing.

        Args:
            adsh: Accession number

        Returns:
            Dictionary with 'submission', 'numbers', and 'text' DataFrames
        """
        result = {}

        # Submission info
        sub_df = self.load_submissions()
        result['submission'] = sub_df[sub_df['adsh'] == adsh].copy()

        # Numerical facts
        num_df = self.load_numbers()
        result['numbers'] = num_df[num_df['adsh'] == adsh].copy()

        # Text facts
        txt_df = self.load_text()
        result['text'] = txt_df[txt_df['adsh'] == adsh].copy()

        return result

    def get_company_financials(self, cik: Union[str, int],
                              tags: Optional[List[str]] = None,
                              forms: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get financial data for a company.

        Args:
            cik: Central Index Key
            tags: List of XBRL tags to filter (e.g., ['Assets', 'Revenues'])
                 If None, returns all tags
            forms: List of form types to filter (e.g., ['10-K', '10-Q'])
                  If None, returns all forms

        Returns:
            DataFrame with company's financial data
        """
        # Get company submissions
        submissions = self.get_company_submissions(cik)

        if submissions.empty:
            return pd.DataFrame()

        # Filter by form type if specified
        if forms:
            submissions = submissions[submissions['form'].isin(forms)]

        # Get accession numbers
        adshs = submissions['adsh'].unique()

        # Get numerical data for these submissions
        num_df = self.load_numbers()
        company_data = num_df[num_df['adsh'].isin(adshs)].copy()

        # Filter by tags if specified
        if tags:
            company_data = company_data[company_data['tag'].isin(tags)]

        # Merge with submission info
        result = company_data.merge(
            submissions[['adsh', 'name', 'form', 'period', 'filed']],
            on='adsh',
            how='left'
        )

        return result

    def get_tag_info(self, tag: str) -> pd.DataFrame:
        """
        Get information about a specific XBRL tag.

        Args:
            tag: XBRL tag name

        Returns:
            DataFrame with tag information
        """
        tag_df = self.load_tags()
        return tag_df[tag_df['tag'] == tag].copy()

    def search_tags(self, keyword: str) -> pd.DataFrame:
        """
        Search for tags by keyword in label or description.

        Args:
            keyword: Keyword to search for

        Returns:
            DataFrame with matching tags
        """
        tag_df = self.load_tags()

        # Search in tag name and label
        mask = (
            tag_df['tag'].str.contains(keyword, case=False, na=False) |
            tag_df['tlabel'].str.contains(keyword, case=False, na=False)
        )

        return tag_df[mask][['tag', 'version', 'tlabel', 'datatype']].drop_duplicates('tag')

    def get_financial_summary(self, cik: Union[str, int],
                             form: str = '10-K') -> pd.DataFrame:
        """
        Get a summary of key financial metrics for a company.

        Args:
            cik: Central Index Key
            form: Form type (default: '10-K' for annual reports)

        Returns:
            DataFrame with key financial metrics over time
        """
        # Common financial statement tags
        key_tags = [
            'Assets', 'AssetsCurrent', 'Liabilities', 'LiabilitiesCurrent',
            'StockholdersEquity', 'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
            'CostOfRevenue', 'GrossProfit', 'OperatingIncomeLoss',
            'NetIncomeLoss', 'EarningsPerShareBasic', 'EarningsPerShareDiluted',
            'CashAndCashEquivalentsAtCarryingValue', 'PropertyPlantAndEquipmentNet'
        ]

        financials = self.get_company_financials(cik, tags=key_tags, forms=[form])

        if financials.empty:
            return pd.DataFrame()

        # Pivot to create a summary table
        summary = financials.pivot_table(
            index='period',
            columns='tag',
            values='value',
            aggfunc='first'  # Take first value if duplicates
        )

        summary = summary.sort_index(ascending=False)

        return summary

    def export_to_csv(self, df: pd.DataFrame, output_path: Union[str, Path]):
        """
        Export DataFrame to CSV.

        Args:
            df: DataFrame to export
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        print(f"Exported to: {output_path}")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the loaded data.

        Returns:
            Dictionary with counts and statistics
        """
        stats = {}

        if self.sub_file.exists():
            sub_df = self.load_submissions()
            stats['total_submissions'] = len(sub_df)
            stats['unique_companies'] = sub_df['cik'].nunique()
            stats['form_types'] = sub_df['form'].nunique()

        if self.num_file.exists():
            num_df = self.load_numbers()
            stats['numerical_facts'] = len(num_df)
            stats['unique_tags'] = num_df['tag'].nunique()

        if self.tag_file.exists():
            tag_df = self.load_tags()
            stats['tag_definitions'] = len(tag_df)

        if self.txt_file.exists():
            txt_df = self.load_text()
            stats['text_facts'] = len(txt_df)

        return stats

    def get_company_sic_codes(self, ciks: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get SIC codes for companies from submissions data.

        The SIC (Standard Industrial Classification) code is used to classify
        companies by industry sector for peer comparison analysis.

        Args:
            ciks: Optional list of CIKs to filter (if None, returns all)

        Returns:
            DataFrame with columns: cik, company_name, sic_code
        """
        sub_df = self.load_submissions()

        # Get unique company-SIC mappings
        # Use most recent filing's SIC code per company (sorted by filed date)
        company_sic = sub_df.sort_values('filed', ascending=False).drop_duplicates(
            subset=['cik'], keep='first'
        )[['cik', 'name', 'sic']].copy()

        company_sic.columns = ['cik', 'company_name', 'sic_code']

        # Pad CIK to 10 digits (standard SEC format)
        company_sic['cik'] = company_sic['cik'].apply(
            lambda x: str(int(x)).zfill(10) if pd.notna(x) else ''
        )

        # Filter by specific CIKs if provided
        if ciks is not None:
            # Normalize input CIKs to 10-digit format
            ciks_normalized = [str(c).zfill(10) for c in ciks]
            company_sic = company_sic[company_sic['cik'].isin(ciks_normalized)]

        return company_sic

    def get_companies_by_sic(self, sic_prefix: str) -> pd.DataFrame:
        """
        Get all companies with SIC codes starting with given prefix.

        Useful for finding all companies in a specific industry sector.

        Args:
            sic_prefix: SIC code prefix (e.g., '73' for Business Services,
                       '36' for Electronic Equipment)

        Returns:
            DataFrame with matching companies
        """
        sub_df = self.load_submissions()

        # Filter by SIC prefix
        mask = sub_df['sic'].astype(str).str.startswith(str(sic_prefix))

        result = sub_df[mask][['cik', 'name', 'sic', 'form', 'period']].drop_duplicates('cik')

        # Format CIK
        result = result.copy()
        result['cik'] = result['cik'].apply(
            lambda x: str(int(x)).zfill(10) if pd.notna(x) else ''
        )

        return result


# Example usage
if __name__ == "__main__":
    # Initialize parser with a quarterly data directory
    parser = SECDataParser("./sec_data/2024q4")

    # Get statistics
    print("Data Statistics:")
    stats = parser.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    print("\n" + "="*60 + "\n")

    # Search for a company
    print("Searching for Apple...")
    companies = parser.get_company_by_name("Apple")
    print(companies[['cik', 'name']].head())

    # Get Apple's data (CIK: 320193)
    apple_cik = "0000320193"
    print(f"\nApple's submissions:")
    submissions = parser.get_company_submissions(apple_cik)
    print(submissions[['form', 'period', 'filed']].head())

    # Get financial summary
    print(f"\nApple's financial summary:")
    summary = parser.get_financial_summary(apple_cik)
    print(summary.head())

    # Search for revenue-related tags
    print("\nSearching for 'revenue' tags:")
    revenue_tags = parser.search_tags("revenue")
    print(revenue_tags.head(10))
