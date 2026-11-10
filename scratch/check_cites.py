import re

with open('docs/manuscript.tex') as f:
    text = f.read()

cites = set()
for c in re.findall(r'\\cite\{([^}]+)\}', text):
    for item in c.split(','):
        if item.strip():
            cites.add(item.strip())

bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', text))

print(f'Total citations in text: {len(cites)}')
print(f'Total bibitems in manuscript: {len(bibitems)}')

missing_in_bibitems = cites - bibitems
print('Citations missing in bibitems:', missing_in_bibitems)

with open('docs/cas-refs.bib') as f:
    bib_text = f.read()

bib_keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib_text))
print(f'Total keys in cas-refs.bib: {len(bib_keys)}')
missing_in_bibfile = cites - bib_keys
print('Citations missing in cas-refs.bib:', missing_in_bibfile)
