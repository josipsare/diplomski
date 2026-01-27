"""
SEC Financial Statement Data Sets Downloader

Downloads quarterly financial statement data sets from the SEC EDGAR database.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Optional, List
import time
from datetime import datetime


class SECDataDownloader:
    """
    Download SEC financial statement data sets.

    The SEC provides quarterly ZIP files containing financial data from XBRL filings.
    Each ZIP contains SUB.txt, NUM.txt, TAG.txt, and TXT.txt files.
    """

    BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"

    def __init__(self, user_agent: str, output_dir: str = "./sec_data"):
        """
        Initialize the downloader.

        Args:
            user_agent: User-Agent string (REQUIRED by SEC).
                       Format: "CompanyName Contact@email.com"
            output_dir: Directory to save downloaded files
        """
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "user_agent is required and must include an email address.\n"
                "Example: 'MyCompany admin@example.com'"
            )

        self.user_agent = user_agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SEC requests max 10 requests/second - we'll be more conservative
        self.request_delay = 0.15  # ~6-7 requests per second

    def _get_headers(self) -> dict:
        """Get HTTP headers with required User-Agent."""
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }

    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from the SEC.

        Args:
            url: URL to download
            output_path: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Downloading: {url}")
            response = requests.get(url, headers=self._get_headers(), stream=True)
            response.raise_for_status()

            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            # print(f"  Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='\r')

            print(f"\n  Downloaded: {output_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  File not found: {url}")
            else:
                print(f"  HTTP Error: {e}")
            return False
        except Exception as e:
            print(f"  Error downloading: {e}")
            return False

    def _extract_zip(self, zip_path: Path, extract_dir: Path) -> bool:
        """
        Extract ZIP file.

        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract to

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Extracting: {zip_path.name}")
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # List extracted files
            extracted_files = list(extract_dir.glob("*.txt"))
            print(f"  Extracted {len(extracted_files)} files:")
            for f in extracted_files:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"    - {f.name} ({size_mb:.1f} MB)")

            return True

        except Exception as e:
            print(f"  Error extracting: {e}")
            return False

    def download_quarter(self, year: int, quarter: int, extract: bool = True,
                        keep_zip: bool = False) -> Optional[Path]:
        """
        Download a single quarter of data.

        Args:
            year: Year (e.g., 2024)
            quarter: Quarter (1-4)
            extract: Whether to extract the ZIP file
            keep_zip: Whether to keep the ZIP file after extraction

        Returns:
            Path to extracted directory if successful, None otherwise
        """
        if quarter not in [1, 2, 3, 4]:
            raise ValueError("Quarter must be 1, 2, 3, or 4")

        filename = f"{year}q{quarter}.zip"
        url = f"{self.BASE_URL}/{filename}"

        zip_path = self.output_dir / filename
        extract_dir = self.output_dir / f"{year}q{quarter}"

        # Check if already downloaded
        if extract_dir.exists() and list(extract_dir.glob("*.txt")):
            print(f"Data already exists: {extract_dir}")
            return extract_dir

        # Download
        if not zip_path.exists():
            success = self._download_file(url, zip_path)
            if not success:
                return None
            time.sleep(self.request_delay)  # Rate limiting
        else:
            print(f"ZIP already exists: {zip_path}")

        # Extract
        if extract:
            success = self._extract_zip(zip_path, extract_dir)
            if not success:
                return None

            # Remove ZIP if requested
            if not keep_zip:
                zip_path.unlink()
                print(f"  Removed ZIP file: {zip_path.name}")

        return extract_dir

    def download_year(self, year: int, extract: bool = True,
                     keep_zip: bool = False) -> List[Path]:
        """
        Download all quarters for a year.

        Args:
            year: Year (e.g., 2024)
            extract: Whether to extract the ZIP files
            keep_zip: Whether to keep ZIP files after extraction

        Returns:
            List of paths to extracted directories
        """
        print(f"\n{'='*60}")
        print(f"Downloading SEC data for {year}")
        print(f"{'='*60}\n")

        extracted_dirs = []

        for quarter in [1, 2, 3, 4]:
            print(f"\n--- Quarter {quarter} ---")
            result = self.download_quarter(year, quarter, extract, keep_zip)
            if result:
                extracted_dirs.append(result)
            time.sleep(self.request_delay)  # Rate limiting between quarters

        print(f"\n{'='*60}")
        print(f"Downloaded {len(extracted_dirs)}/4 quarters for {year}")
        print(f"{'='*60}\n")

        return extracted_dirs

    def download_range(self, start_year: int, end_year: int,
                      extract: bool = True, keep_zip: bool = False) -> List[Path]:
        """
        Download multiple years of data.

        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            extract: Whether to extract the ZIP files
            keep_zip: Whether to keep ZIP files after extraction

        Returns:
            List of paths to extracted directories
        """
        all_dirs = []

        for year in range(start_year, end_year + 1):
            dirs = self.download_year(year, extract, keep_zip)
            all_dirs.extend(dirs)

        return all_dirs

    def get_latest_quarter(self) -> tuple[int, int]:
        """
        Get the latest available quarter based on current date.

        Returns:
            Tuple of (year, quarter)
        """
        now = datetime.now()
        year = now.year
        month = now.month

        # SEC data is typically 1-2 months behind
        # Adjust for delay
        if month <= 3:
            return (year - 1, 4)
        elif month <= 6:
            return (year, 1)
        elif month <= 9:
            return (year, 2)
        else:
            return (year, 3)

    def download_latest(self, extract: bool = True,
                       keep_zip: bool = False) -> Optional[Path]:
        """
        Download the latest available quarter.

        Args:
            extract: Whether to extract the ZIP file
            keep_zip: Whether to keep ZIP file after extraction

        Returns:
            Path to extracted directory if successful
        """
        year, quarter = self.get_latest_quarter()
        print(f"Latest estimated quarter: {year}Q{quarter}")
        return self.download_quarter(year, quarter, extract, keep_zip)


# Example usage
if __name__ == "__main__":
    # IMPORTANT: Replace with your information
    USER_AGENT = "Josip Sare josip.sare@gmail.com"

    downloader = SECDataDownloader(
        user_agent=USER_AGENT,
        output_dir="./sec_data"
    )

    # Download latest quarter
    print("Downloading latest quarter...")
    downloader.download_latest()

    # Or download specific quarter
    # downloader.download_quarter(2024, 4)

    # Or download entire year
    # downloader.download_year(2024)

    # Or download range
    # downloader.download_range(2023, 2024)
