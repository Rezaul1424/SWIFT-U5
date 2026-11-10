import re

with open('docs/manuscript.tex', 'r') as f:
    text = f.read()

print('Inputs:', re.findall(r'\\input\{([^}]+)\}', text))
print('Includes:', re.findall(r'\\include\{([^}]+)\}', text))
print('Graphics:', re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text))
print('Casinlinefig:', re.findall(r'\\casinlinefig(?:tall)?(?:\[[^\]]*\])?\{([^}]+)\}', text))
print('Bib:', re.findall(r'\\bibliography\{([^}]+)\}', text))
print('Documentclass:', re.findall(r'\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}', text))
