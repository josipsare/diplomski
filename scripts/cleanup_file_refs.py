"""Final cleanup pass: remove residual file references and broken artifacts."""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

p = Path('thesis/diplomski.tex')
text = p.read_text(encoding='utf-8')

# Fix specific residue
text = text.replace(
    'Rezultati svakog detektora pohranjuju se u .parquet} sa shemom \\texttt{(cik, period\\_end, score)}.',
    'Rezultati svakog detektora pohranjuju se u jedinstvenoj tablici sa shemom (CIK firme, datum perioda, anomalijska ocjena).'
)

# Catch any remaining .parquet} fragments
text = re.sub(r'\bu \.parquet\}', 'u jedinstvenu tablicu', text)
text = re.sub(r'\.parquet\}', '', text)

# Convert code identifier references like \texttt{fit(X)} etc to plain prose
text = re.sub(r'\\texttt\{fit\(X\)\}', '\\\\emph{fit}', text)
text = re.sub(r'\\texttt\{score\(X\)\}', '\\\\emph{score}', text)
text = re.sub(r'\\texttt\{ndarray\}', 'numerički niz', text)
text = re.sub(r'\\texttt\{torch\\\\_manual\\_seed\(\d+\)\}', 'fiksna sjemenka', text)
text = re.sub(r'\\texttt\{np\\\\_random\\_seed\(\d+\)\}', 'fiksna sjemenka', text)
text = re.sub(r'\\texttt\{torch\.manual\\_seed\([^)]*\)\}', 'fiksna PyTorch sjemenka', text)
text = re.sub(r'\\texttt\{np\.random\.seed\([^)]*\)\}', 'fiksna NumPy sjemenka', text)

# Remove citation of CIK column refs
text = re.sub(r'\\texttt\{\(cik, period\\_end, score\)\}', '(CIK firme, datum perioda, ocjena)', text)
text = re.sub(r'\\texttt\{ddate\}', '\\\\emph{ddate}', text)
text = re.sub(r'\\texttt\{qtrs\}', '\\\\emph{qtrs}', text)
text = re.sub(r'\\texttt\{form\}', '\\\\emph{form}', text)
text = re.sub(r'\\texttt\{filed\}', '\\\\emph{filed}', text)
text = re.sub(r'\\texttt\{yfinance\}', '\\\\emph{yfinance}', text)
text = re.sub(r'\\texttt\{revenue\\_growth\}', '\\\\emph{revenue\\\\_growth}', text)
text = re.sub(r'\\texttt\{asset\\_growth\}', '\\\\emph{asset\\\\_growth}', text)
text = re.sub(r'\\texttt\{10-K/A\}', '10-K/A', text)
text = re.sub(r'\\texttt\{10-Q/A\}', '10-Q/A', text)

# Generic remaining \texttt{X} where X is a likely path component
text = re.sub(r'\\texttt\{(Assets|Liabilities|Revenues|NetIncomeLoss|CashAndCashEquivalentsAtCarryingValue|LongTermDebt|OperatingIncomeLoss|NetCashProvidedByUsedInOperatingActivities|IncreaseDecreaseInAccountsReceivable)\}',
              r'\\emph{\1}', text)

# Remove orphan empty parens (paths removed)
text = re.sub(r'\(\s*\)', '', text)
text = re.sub(r'\s+\.', '.', text)

# Double-comma cleanup
text = re.sub(r',\s*,', ',', text)
text = re.sub(r',\s*\.', '.', text)
text = re.sub(r' {2,}', ' ', text)

p.write_text(text, encoding='utf-8')

# Verify no remaining file-paths
remaining = re.findall(r'\\texttt\{[^}]*\.(?:py|tex|bib|parquet|cls|csv|npz|md|json|txt)\}', text)
print(f"Remaining file-extension texttt: {len(remaining)}")
for r in remaining[:5]:
    print(f"  {r}")
remaining_paths = re.findall(r'\\texttt\{(?:src|data|scripts|legacy|thesis|sec_data)/[^}]*\}', text)
print(f"Remaining path texttt: {len(remaining_paths)}")
for r in remaining_paths[:5]:
    print(f"  {r}")
