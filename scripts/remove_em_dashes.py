"""Remove em-dashes from thesis LaTeX file.

Replacement rules:
1. Item with emphasized X followed by em-dash and description: replace em-dash with colon
2. Same for textbf
3. Section headings with em-dash: replace em-dash with colon
4. General em-dash in prose: replace with comma
5. Unicode em-dash: replace with comma

Keeps en-dashes (two hyphens) which are used for ranges like 2014--2024.
"""

import re
import sys
from pathlib import Path

path = Path('thesis/diplomski.tex')
text = path.read_text(encoding='utf-8')
orig_len = len(text)

# Rule 1: \item \emph{X} --- desc -> \item \emph{X}: desc
text = re.sub(r'(\\item \\emph\{[^}]+\}) --- ', r'\1: ', text)

# Rule 2: \item \textbf{X} --- desc -> \item \textbf{X}: desc
text = re.sub(r'(\\item \\textbf\{[^}]+\}) --- ', r'\1: ', text)

# Rule 3: \subsubsection{X --- Y} -> \subsubsection{X: Y}
text = re.sub(r'(\\subsubsection\{[^}]+) --- ', r'\1: ', text)
text = re.sub(r'(\\subsection\{[^}]+) --- ', r'\1: ', text)
text = re.sub(r'(\\section\{[^}]+) --- ', r'\1: ', text)

# Rule 4: General em-dash with spaces -> comma + space in prose
text = re.sub(r' --- ', ', ', text)

# Rule 5: Unicode em-dash
text = text.replace(' — ', ', ')
text = text.replace('—', ',')

# Cleanup: double commas
text = re.sub(r',\s*,', ',', text)
# Cleanup: comma before period
text = re.sub(r',\s*\.', '.', text)

path.write_text(text, encoding='utf-8')

sys.stdout.reconfigure(encoding='utf-8')
print(f'Processed: {orig_len} -> {len(text)} chars')
print(f'Remaining triple-dash count: {text.count("---")}')
print(f'Remaining unicode em-dash count: {text.count(chr(0x2014))}')
