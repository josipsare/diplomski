"""Apply FER UniZG LaTeX template to existing diplomski.tex.

Transformations:
1. Replace preamble + manual title page + abstract + TOC with FER template
2. Convert section hierarchy IN REVERSE ORDER:
     \\section{X}        -> \\chapter{X}
     \\subsection{X}     -> \\section{X}
     \\subsubsection{X}  -> \\subsection{X}
   (Reverse order so converted strings don't get re-promoted.)
3. Remove all file/folder/code path references (texttt with file extensions or paths)
4. Remove 'Tehnicka reproducibilnost' and 'Hiperparametri' appendices (full of file paths)
5. Add bilingual sazetak/abstract/kljucnerijeci/keywords at end
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

src_path = Path('thesis/diplomski.tex')
text = src_path.read_text(encoding='utf-8')

# ----- Phase 1: Extract body (from \section{Uvod} to before \end{document}) -----
m_intro = re.search(r'\\section\{Uvod\}', text)
m_end = re.search(r'\\end\{document\}', text)
if not m_intro or not m_end:
    raise RuntimeError("Could not find Uvod or document end")

body = text[m_intro.start():m_end.start()]
print(f"Phase 1: body extracted, {len(body)} chars")

# ----- Phase 2: Hierarchy conversion (REVERSE ORDER to avoid double-promotion) -----
# Process top-level FIRST so converted strings don't get re-matched
body = re.sub(r'\\section\*?\{', '\\\\chapter{', body)
body = re.sub(r'\\subsection\*?\{', '\\\\section{', body)
body = re.sub(r'\\subsubsection\*?\{', '\\\\subsection{', body)
print("Phase 2: hierarchy converted (section->chapter, subsection->section, subsubsection->subsection)")

# ----- Phase 3: Remove file/folder/code references -----

# A) Specific known patterns FIRST (multi-arg lists, etc.)
# "kod je strukturiran s razdvojenim slojevima (\texttt{src/data}, \texttt{src/features}, ...)"
body = re.sub(
    r'kod je strukturiran s razdvojenim slojevima \(\\texttt\{[^}]+\}(?:,\s*\\texttt\{[^}]+\})*\)',
    'kod je strukturiran s razdvojenim slojevima u jasno odvojene cjeline za pripremu podataka, izvedene značajke, NN detektore, NN regresor utjecaja i evaluaciju',
    body
)

# "premješteni u \texttt{legacy/} folder" -> "premješteni iz primarnog narativa"
body = re.sub(r'premje[sš]teni u\s*\\texttt\{[^}]+\}\s*folder', 'premješteni iz primarnog narativa', body)
body = re.sub(r'premje[sš]teni u\s*\\texttt\{[^}]+\}', 'premješteni iz primarnog narativa', body)

# B) Specific full sentences containing file paths
# "Svaki detektor implementira identičnu shemu (\texttt{src/models/base.py}):"
body = re.sub(
    r'Svaki detektor implementira identi[čc]nu shemu\s*\(\\texttt\{[^}]+\}\):',
    'Svaki detektor implementira identičnu jedinstvenu sheme:',
    body
)

# "Rezultati svakog detektora pohranjuju se u \texttt{data/output/scores/\{model\_name\}.parquet} sa shemom \texttt{(cik, period\_end, score)}"
body = re.sub(
    r'Rezultati svakog detektora pohranjuju se u\s*\\texttt\{[^}]+\}\s*sa shemom\s*\\texttt\{[^}]+\}',
    'Rezultati svakog detektora pohranjuju se u jedinstvenoj tablici sa shemom (CIK firme, datum perioda, anomalijska ocjena)',
    body
)

# "jedinstvena evaluacijska skripta (\texttt{eval\_nn\_detectors.py})"
body = re.sub(
    r'jedinstvena evaluacijska skripta\s*\(\\texttt\{[^}]+\}\)',
    'jedinstvena evaluacijska procedura',
    body
)

# "Klasične metode (event study, ...) implementirane su u \texttt{legacy/classical\_impact/} folderu kao reference"
body = re.sub(
    r'implementirane su u\s*\\texttt\{[^}]+\}\s*folderu kao reference',
    'razrađene su kao reference izvan glavnog narativa',
    body
)

# "Implementirano je kao referenca u \texttt{legacy/baselines/ensemble.py} ali nije..."
body = re.sub(
    r'Implementirano je kao referenca u\s*\\texttt\{[^}]+\}\s*ali nije',
    'Pristup je razmotren kao referenca, ali nije',
    body
)

# "panel_quarterly.parquet" / "panel_annual.parquet" in prose: replace with descriptive name
body = re.sub(r'Rezultat:\s*\\texttt\{panel_quarterly\.parquet\}', 'Rezultat: kvartalni panel', body)
body = re.sub(r'\\texttt\{panel_quarterly\.parquet\}', 'kvartalnom panelu', body)
body = re.sub(r'godišnji panel\s*\\texttt\{panel_annual\.parquet\}', 'godišnji panel', body)
body = re.sub(r'\\texttt\{panel_annual\.parquet\}', 'godišnjem panelu', body)

# C) Generic catchalls: any \texttt{<path>/<...>} or \texttt{<name>.<ext>}
body = re.sub(r'\\texttt\{[a-zA-Z0-9_/\\.-]+\.(py|tex|bib|parquet|cls|csv|npz|md|json|txt)\}', '', body)
body = re.sub(r'\\texttt\{(src|data|scripts|legacy|thesis|sec_data|figures)/[^}]*\}', '', body)

# D) Cleanup: orphan parens/commas left behind
body = re.sub(r'\(\s*\)', '', body)
body = re.sub(r'\(\s*,\s*\)', '', body)
body = re.sub(r'\s*\(\s*,*\s*\)\s*', ' ', body)
body = re.sub(r',\s*,', ',', body)
body = re.sub(r',\s*\.', '.', body)
body = re.sub(r'\s+:', ':', body)
body = re.sub(r'  +', ' ', body)

print("Phase 3: file/code references removed")

# ----- Phase 4: Remove file-path-heavy appendices -----
def remove_chapter(text, title_pattern):
    pat = r'\\chapter\{' + title_pattern + r'\}.*?(?=\\chapter\{|\Z)'
    return re.sub(pat, '', text, flags=re.DOTALL)

body = remove_chapter(body, 'Tehni[čc]ka reproducibilnost')
body = remove_chapter(body, 'Hiperparametri')
print("Phase 4: file-path appendices removed")

# Cleanup newpage/comment markers
body = re.sub(r'\\newpage\s*\n\s*\\chapter', r'\\chapter', body)
body = re.sub(r'^% ={5,}\n.*\n% ={5,}\n', '', body, flags=re.MULTILINE)
body = re.sub(r'\\bibliographystyle\{[^}]+\}\s*\n', '', body)
body = re.sub(r'\\bibliography\{[^}]+\}\s*\n', '', body)

# Split appendix
appendix_marker = '\\appendix'
appendix_idx = body.find(appendix_marker)
if appendix_idx >= 0:
    main_body = body[:appendix_idx]
    appendix_body = body[appendix_idx + len(appendix_marker):]
    print(f"Phase 5: split main vs appendix at {appendix_idx}")
else:
    main_body = body
    appendix_body = ''

# ----- Build new file -----
NEW_PREAMBLE = r"""\documentclass[diplomskirad]{fer}

\usepackage{multirow}
\usepackage{booktabs}


%--- PODACI O RADU / THESIS INFORMATION ----------------------------------------

\title{Identification of Financial Anomalies Using Deep Neural Networks and the Impact of Identified Anomalies on Company Business}

\naslov{Identifikacija financijskih anomalija primjenom dubokih neuronskih mreža te utjecaj identificiranih anomalija na poslovanje kompanije}

\brojrada{}

\author{Josip Sare}

\mentor{}

\date{May, 2026}

\datum{svibanj, 2026.}


\begin{document}


% Naslovnica se automatski generira
\maketitle


% Zadatak se ubacuje iz vanjske PDF datoteke preuzete s FERWeb-a
% \zadatak{zadatak.pdf}


\begin{zahvale}
Zahvaljujem mentoru na strpljenju i smjernicama kroz cijeli proces izrade ovog rada. Zahvaljujem obitelji i prijateljima na podršci.
\end{zahvale}


\mainmatter


\tableofcontents


"""

NEW_BIB_AND_ABSTRACTS = r"""

%--- LITERATURA / REFERENCES ---------------------------------------------------
\bibliography{references}


%--- SAŽETAK / ABSTRACT --------------------------------------------------------

\begin{sazetak}
Ovaj diplomski rad primjenjuje četiri arhitekture dubokih neuronskih mreža: statički autoenkoder (AE), varijacijski autoenkoder (VAE), rekurentni LSTM autoenkoder (LSTM-AE) i Transformer enkoder, na nenadzirano otkrivanje anomalija u financijskim izvještajima 2.499 američkih poduzeća iz indeksa Russell 3000 u razdoblju od jedanaest godina (2014.--2024.). Za svaki detektor izračunata je anomalijska ocjena, koja je zatim evaluirana na dvostrukoj razini: nenadzirana dijagnostika (slaganje među detektorima preko Kendall $\tau$ i Jaccard@100) i nadzirano benchmarkiranje protiv oznaka revidiranih financijskih izvještaja. Empirijski nalaz: sve četiri arhitekture konvergiraju na ROC-AUC od približno 0,63, što ukazuje na strop u ulaznoj modalnosti. Drugi dio rada kvantificira utjecaj identificiranih anomalija na poslovanje kompanije primjenom NN regresora (MLP). Ključni nalazi uz petostruku replikaciju sjemenki i kontrolu za rast firme: volatilnost predvidiva s $\Delta R^2 = +2{,}87 \pm 0{,}80$ p.b.\ (LSTM-AE), godišnji prinos $+0{,}78 \pm 0{,}36$ p.b.\ (AE), rast volumena $+1{,}14 \pm 0{,}36$ p.b.\ (LSTM-AE), maksimalni drawdown $+0{,}49 \pm 0{,}48$ p.b.\ (Transformer). Sporedna metodološka kontribucija je demonstracija da je naivni izračun $\Delta R^2$ bez kontrole za rast precijenjen za 40\,--\,80\,\%.
\end{sazetak}

\begin{kljucnerijeci}
duboke neuronske mreže; nenadzirano učenje; otkrivanje anomalija; financijski izvještaji; autoenkoder; Transformer; LSTM; forenzičko računovodstvo; SEC EDGAR; Russell 3000
\end{kljucnerijeci}


\begin{abstract}
This master's thesis applies four deep neural network architectures, namely a static autoencoder (AE), a variational autoencoder (VAE), a recurrent LSTM autoencoder (LSTM-AE), and a Transformer encoder, to unsupervised anomaly detection in the financial statements of 2,499 U.S. publicly listed companies from the Russell 3000 index over an eleven-year period (2014--2024). For each detector, an anomaly score is computed and evaluated through a dual track: unsupervised diagnostics (inter-detector agreement via Kendall $\tau$ and Jaccard@100) and supervised benchmarking against restatement labels. Empirically, all four architectures converge to a ROC-AUC of approximately 0.63, indicating a ceiling in the input modality. The second part of the thesis quantifies the impact of identified anomalies on company business outcomes via a deep neural regressor (MLP). Key findings using five-seed replication and firm-growth controls: volatility is predictable with $\Delta R^2 = +2.87 \pm 0.80$ percentage points (LSTM-AE), annual stock return $+0.78 \pm 0.36$ p.p.\ (AE), trading-volume growth $+1.14 \pm 0.36$ p.p.\ (LSTM-AE), and maximum drawdown $+0.49 \pm 0.48$ p.p.\ (Transformer). A secondary methodological contribution is the demonstration that naive $\Delta R^2$ estimates without firm-growth controls are inflated by 40\,--\,80\,\%.
\end{abstract}

\begin{keywords}
deep neural networks; unsupervised learning; anomaly detection; financial reporting; autoencoder; Transformer; LSTM; forensic accounting; SEC EDGAR; Russell 3000
\end{keywords}

"""

if appendix_body.strip():
    NEW_APPENDIX = "\n\n%--- PRIVITCI / APPENDIX -------------------------------------------------------\n\\backmatter\n" + appendix_body
else:
    NEW_APPENDIX = ''

new_text = NEW_PREAMBLE + main_body + NEW_APPENDIX + NEW_BIB_AND_ABSTRACTS + "\n\\end{document}\n"

new_text = re.sub(r'\n{4,}', '\n\n\n', new_text)
src_path.write_text(new_text, encoding='utf-8')

print(f"\nFinal file: {len(new_text)} chars")
import subprocess
# Count headers with grep
for tag in ('chapter', 'section', 'subsection'):
    count = len(re.findall(r'\\' + tag + r'\{', new_text))
    print(f"  \\{tag}: {count}")
