import re
from pathlib import Path

def verify_latex(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check environments
    begins = re.findall(r'\\begin\{([^}]+)\}', content)
    ends = re.findall(r'\\end\{([^}]+)\}', content)
    
    print(f"Total \\begin: {len(begins)}, Total \\end: {len(ends)}")
    
    stack = []
    for line_num, line in enumerate(content.splitlines(), 1):
        # strip comments
        line_no_comment = re.sub(r'(?<!\\)%.*$', '', line)
        for m in re.finditer(r'\\begin\{([^}]+)\}', line_no_comment):
            stack.append((m.group(1), line_num))
        for m in re.finditer(r'\\end\{([^}]+)\}', line_no_comment):
            env_end = m.group(1)
            if not stack:
                print(f"ERROR: Extra \\end{{{env_end}}} at line {line_num}")
            else:
                env_begin, begin_line = stack.pop()
                if env_begin != env_end:
                    print(f"MISMATCH: \\begin{{{env_begin}}} at line {begin_line} closed by \\end{{{env_end}}} at line {line_num}")

    if stack:
        print(f"ERROR: Unclosed environments: {stack}")
    else:
        print("[PASS] All LaTeX environments properly matched!")

    # Check figure files
    fig_dir = Path("docs/figures")
    for fig in re.findall(r'\\casinlinefig(?:tall)?(?:\[[^\]]*\])?\{([^}]+)\}', content):
        if not (fig_dir / fig).exists():
            print(f"ERROR: Figure {fig} not found in {fig_dir}!")
        else:
            print(f"[PASS] Figure {fig} verified.")

    # Check bibitems
    cites = set()
    for c in re.findall(r'\\cite\{([^}]+)\}', content):
        for item in c.split(','):
            if item.strip():
                cites.add(item.strip())
    bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', content))
    missing_cites = cites - bibitems
    if missing_cites:
        print(f"ERROR: Missing bibitems for citations: {missing_cites}")
    else:
        print(f"[PASS] All {len(cites)} citations resolved in thebibliography ({len(bibitems)} total bibitems)!")

verify_latex('docs/manuscript.tex')
