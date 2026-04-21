import sys
import subprocess

try:
    import pypdf
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pypdf', '-q'])
    import pypdf

reader = pypdf.PdfReader('NeuroTech BR4IN.IO Hackathon Software Projects Specification (1).pdf')
with open('.tmp/pdf_out.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(reader.pages):
        f.write(f'--- Page {i+1} ---\n')
        f.write(p.extract_text() + '\n')
