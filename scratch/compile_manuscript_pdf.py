#!/usr/bin/env python3
"""
Compile docs/manuscript.tex into a publication-quality manuscript PDF (docs/manuscript.pdf)
using ReportLab, with full figures, tables, algorithms, and references.
"""
import re
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Image
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register fonts
pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', '/System/Library/Fonts/Supplemental/Arial Italic.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf'))
pdfmetrics.registerFont(TTFont('ArialUnicode', '/Library/Fonts/Arial Unicode.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-BoldItalic')

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Arial', 7.5)
        self.setFillColor(colors.HexColor('#64748B'))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 812, "SWIFT-U5 | Survey-Weighted Child Mortality Risk Stratification in Bangladesh")
            self.drawRightString(559, 812, "Chowdhury et al.")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.5)
            self.line(36, 806, 559, 806)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(36, 36, 559, 36)
        self.drawString(36, 25, "Faculty Research Manuscript • BDHS 2004–2022 • Method-Based Article")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 25, page_text)
        self.restoreState()

styles = {
    'Title': ParagraphStyle('DocTitle', fontName='Arial-Bold', fontSize=14, leading=17, textColor=colors.HexColor('#0F294A'), alignment=1, spaceAfter=8),
    'Authors': ParagraphStyle('DocAuthors', fontName='Arial-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=2),
    'Affil': ParagraphStyle('DocAffil', fontName='Arial-Italic', fontSize=7.2, leading=9.5, textColor=colors.HexColor('#475569'), alignment=1, spaceAfter=10),
    'AbstractHead': ParagraphStyle('AbsHead', fontName='Arial-Bold', fontSize=9, leading=11, textColor=colors.HexColor('#0F294A'), spaceBefore=4, spaceAfter=3),
    'Abstract': ParagraphStyle('AbsText', fontName='ArialUnicode', fontSize=7.2, leading=10, textColor=colors.HexColor('#1E293B'), spaceAfter=6),
    'Keywords': ParagraphStyle('KwText', fontName='ArialUnicode', fontSize=7.0, leading=9.5, textColor=colors.HexColor('#334155'), spaceAfter=10),
    'H1': ParagraphStyle('Heading1', fontName='Arial-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#0F294A'), spaceBefore=10, spaceAfter=4, keepWithNext=True),
    'H2': ParagraphStyle('Heading2', fontName='Arial-Bold', fontSize=9, leading=11.5, textColor=colors.HexColor('#1E3A8A'), spaceBefore=7, spaceAfter=3, keepWithNext=True),
    'H3': ParagraphStyle('Heading3', fontName='Arial-Bold', fontSize=8, leading=10.5, textColor=colors.HexColor('#2563EB'), spaceBefore=5, spaceAfter=2, keepWithNext=True),
    'Body': ParagraphStyle('BodyTextCustom', fontName='ArialUnicode', fontSize=7.2, leading=10.2, textColor=colors.HexColor('#1E293B'), spaceAfter=3),
    'Bullet': ParagraphStyle('BulletCustom', fontName='ArialUnicode', fontSize=7.0, leading=9.8, textColor=colors.HexColor('#1E293B'), leftIndent=12, spaceAfter=2),
    'Equation': ParagraphStyle('EquationCustom', fontName='ArialUnicode', fontSize=7.5, leading=10, textColor=colors.HexColor('#0F294A'), alignment=1, spaceBefore=3, spaceAfter=3),
    'Caption': ParagraphStyle('FigCaption', fontName='Arial-Bold', fontSize=7.0, leading=9.2, textColor=colors.HexColor('#334155'), alignment=1, spaceBefore=3, spaceAfter=6),
    'TableFoot': ParagraphStyle('TFoot', fontName='ArialUnicode', fontSize=6.2, leading=8.2, textColor=colors.HexColor('#475569'), spaceBefore=2, spaceAfter=4),
}

import xml.sax.saxutils as saxutils

def clean_latex(s):
    # Remove latex comments
    s = re.sub(r'%.*$', '', s)
    # LaTeX typographical replacements
    s = s.replace(r'---', '—').replace(r'--', '–')
    s = s.replace(r'\{', '{').replace(r'\}', '}')
    s = s.replace(r'\%', '%').replace(r'\&', '&')
    s = s.replace(r'\,', ' ').replace(r'\ ', ' ')
    s = s.replace(r'\ge', '≥').replace(r'\le', '≤').replace(r'\times', '×').replace(r'\approx', '≈')
    s = s.replace(r'\cdot', '·').replace(r'\to', '→').replace(r'\pm', '±')
    s = s.replace(r'\in', '∈').replace(r'\Delta', 'Δ').replace(r'\ne', '≠')
    s = s.replace(r'\tau', 'τ').replace(r'\alpha', 'α').replace(r'\beta', 'β').replace(r'\mu', 'μ')
    s = s.replace(r'\sigma', 'σ').replace(r'\lambda', 'λ').replace(r'\pi', 'π')
    s = s.replace(r'\sim', '~').replace(r'\quad', ' ')
    
    # Specific number formats
    s = s.replace(r'1{,}312', '1,312').replace(r'13{,}119', '13,119').replace(r'51{,}489', '51,489')
    s = s.replace(r'2{,}338', '2,338').replace(r'6{,}908', '6,908').replace(r'6{,}150', '6,150')
    s = s.replace(r'8{,}742', '8,742').replace(r'7{,}886', '7,886').replace(r'8{,}684', '8,684')
    s = s.replace(r'29{,}686', '29,686').replace(r'38{,}370', '38,370').replace(r'1{,}000', '1,000')
    s = s.replace(r'13{,}169.9', '13,169.9').replace(r'1{,}260.0', '1,260.0')
    s = s.replace(r'3{,}740.6', '3,740.6').replace(r'3{,}947', '3,947').replace(r'8{,}985.7', '8,985.7')
    s = s.replace(r'8{,}764', '8,764').replace(r'1{,}163.0', '1,163.0').replace(r'1{,}205', '1,205')
    s = s.replace(r'11{,}612.6', '11,612.6').replace(r'11{,}506', '11,506')
    s = s.replace(r'3{,}925.9', '3,925.9').replace(r'4{,}154', '4,154')
    s = s.replace(r'2{,}910', '2,910').replace(r'2{,}604', '2,604').replace(r'2{,}551', '2,551')
    s = s.replace(r'2{,}518', '2,518').replace(r'2{,}536', '2,536').replace(r'3{,}092', '3,092')
    s = s.replace(r'6{,}788', '6,788').replace(r'2{,}415', '2,415')
    
    # Citations and cross-references
    s = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', s)
    s = re.sub(r'\\ref\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\eqref\{([^}]+)\}', r'(\1)', s)
    
    # Iteratively replace formatting from inner to outer
    for _ in range(4):
        s = re.sub(r'\\textbf\{([^{}]+)\}', r'<b>\1</b>', s)
        s = re.sub(r'\\textit\{([^{}]+)\}', r'<i>\1</i>', s)
        s = re.sub(r'\\texttt\{([^{}]+)\}', r'<font face="Courier">\1</font>', s)
        s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s)
        s = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', s)
        s = re.sub(r'\\mathbf\{([^{}]+)\}', r'<b>\1</b>', s)
        s = re.sub(r'\\boldsymbol\{([^{}]+)\}', r'<b>\1</b>', s)
        s = re.sub(r'\\textsc\{([^{}]+)\}', r'\1', s)

    # Subscripts and superscripts
    s = re.sub(r'_\{([^{}]+)\}', r'<sub>\1</sub>', s)
    s = re.sub(r'\^\{([^{}]+)\}', r'<super>\1</super>', s)
    
    # Inline math
    s = re.sub(r'\$([^$]+)\$', r'<i>\1</i>', s)
    
    # Escape standalone & that is not an entity
    s = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', s)
    
    # Remove remaining unmatched curly braces or backslashes
    s = s.replace('{', '').replace('}', '').replace('\\', '')
    return s.strip()

def safe_paragraph(text, style):
    if not text or not str(text).strip():
        return None
    t = clean_latex(str(text))
    if not t:
        return None
    try:
        return Paragraph(t, style)
    except Exception:
        # Strip all xml-like tags and HTML escape for total robustness
        plain = re.sub(r'<[^>]+>', '', t)
        escaped = saxutils.escape(plain)
        try:
            return Paragraph(escaped, style)
        except Exception:
            return None

def build_pdf():
    pdf_path = Path("docs/manuscript.pdf")
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )
    
    story = []

    # Title & Metadata
    title_text = "SWIFT-U5: A Survey-Weighted and Temporally Calibrated Machine Learning Framework for Population-Level Child Mortality Risk Stratification in Bangladesh"
    story.append(safe_paragraph(f"<b>{title_text}</b>", styles['Title']))
    
    authors = "Md. Rafiul Azam Chowdhury, Rezaul Mursalin Khan, Fatema Tuz Zohura Rifa, Tanjim Ahmed Irfad, Arafat Hussain Nahid, Ohidujjaman"
    story.append(safe_paragraph(f"<b>{authors}</b>", styles['Authors']))
    
    affil = "Department of Computer Science and Engineering, United International University, Dhaka 1212, Bangladesh"
    story.append(safe_paragraph(f"<i>{affil}</i>", styles['Affil']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1E3A8A'), spaceBefore=2, spaceAfter=6))

    # Read LaTeX source to parse sections
    with open('docs/manuscript.tex', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    abstract_text = ""
    keywords_text = ""
    in_abstract = False
    in_keywords = False
    
    for l in lines:
        if r'\begin{abstract}' in l:
            in_abstract = True
            continue
        if r'\end{abstract}' in l:
            in_abstract = False
            continue
        if in_abstract:
            abstract_text += l.strip() + " "
        if r'\begin{keywords}' in l:
            in_keywords = True
            continue
        if r'\end{keywords}' in l:
            in_keywords = False
            continue
        if in_keywords:
            keywords_text += l.strip() + " "

    story.append(safe_paragraph("<b>Abstract</b>", styles['AbstractHead']))
    p_abs = safe_paragraph(abstract_text.strip(), styles['Abstract'])
    if p_abs: story.append(p_abs)
    
    kw_clean = keywords_text.replace(r'\sep', ' • ')
    p_kw = safe_paragraph(f"<b>Keywords:</b> {kw_clean}", styles['Keywords'])
    if p_kw: story.append(p_kw)
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=2, spaceAfter=8))

    # Parse body
    current_sec = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if line.startswith(r'\section{'):
            m = re.search(r'\\section\{([^}]+)\}', line)
            if m:
                sec_title = m.group(1)
                p = safe_paragraph(f"<b>{sec_title}</b>", styles['H1'])
                if p: story.append(p)
            continue

        if line.startswith(r'\subsection{'):
            m = re.search(r'\\subsection\{([^}]+)\}', line)
            if m:
                subsec_title = m.group(1)
                p = safe_paragraph(f"<b>{subsec_title}</b>", styles['H2'])
                if p: story.append(p)
            continue

        if line.startswith(r'\subsubsection{'):
            m = re.search(r'\\subsubsection\{([^}]+)\}', line)
            if m:
                subsubsec_title = m.group(1)
                p = safe_paragraph(f"<b>{subsubsec_title}</b>", styles['H3'])
                if p: story.append(p)
            continue

        if line.startswith(r'\paragraph{'):
            m = re.search(r'\\paragraph\{([^}]+)\}', line)
            if m:
                para_title = m.group(1)
                p = safe_paragraph(f"<b>{para_title}</b>", styles['H3'])
                if p: story.append(p)
            continue

        # Handle Figures
        if r'\casinlinefig' in line:
            # \casinlinefig[\linewidth]{workflow_diagram.png}{Caption}{label}
            fig_match = re.search(r'\\casinlinefig(?:tall)?(?:\[[^\]]*\])?\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}', line)
            if fig_match:
                img_name = fig_match.group(1)
                caption = fig_match.group(2)
                img_path = Path("docs/figures") / img_name
                if not img_path.exists():
                    img_path = Path("outputs/figures") / img_name
                if img_path.exists():
                    story.append(Spacer(1, 4))
                    story.append(Image(str(img_path), width=500, height=220, kind='proportional'))
                    p = safe_paragraph(f"<b>Figure:</b> {caption}", styles['Caption'])
                    if p: story.append(p)
                    story.append(Spacer(1, 4))
            continue

        # Handle Algorithms
        if line.startswith(r'\begin{algorithm}'):
            # gather algorithm lines
            alg_lines = []
            while i < len(lines) and not lines[i].strip().startswith(r'\end{algorithm}'):
                alg_lines.append(lines[i].strip())
                i += 1
            i += 1 # skip end
            
            clean_algs = []
            for al in alg_lines:
                if not al.startswith(r'\caption') and not al.startswith(r'\label') and not al.startswith(r'\begin') and not al.startswith(r'\end'):
                    cl = clean_latex(al)
                    if cl:
                        clean_algs.append(cl)
            alg_text = "<br/>".join(clean_algs)
            alg_p = safe_paragraph(f"<b>Algorithm 1: SWIFT-U5 Complete Framework</b><br/>{alg_text}", styles['Bullet'])
            if alg_p:
                t = Table([[alg_p]], colWidths=[523])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94A3B8')),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 4))
            continue

        # Handle Tables
        if line.startswith(r'\begin{table}'):
            # gather table caption and lines
            caption = ""
            tbl_lines = []
            while i < len(lines) and not lines[i].strip().startswith(r'\end{table}'):
                tl = lines[i].strip()
                if r'\caption{' in tl:
                    cm = re.search(r'\\caption\{([^}]+)\}', tl)
                    if cm:
                        caption = cm.group(1)
                tbl_lines.append(tl)
                i += 1
            i += 1

            # Format caption
            if caption:
                story.append(Spacer(1, 4))
                p = safe_paragraph(f"<b>Table:</b> {caption}", styles['Caption'])
                if p: story.append(p)

            # Extract tabular rows if possible
            rows = []
            in_tab = False
            for tl in tbl_lines:
                if r'\begin{tabular' in tl:
                    in_tab = True
                    continue
                if r'\end{tabular' in tl:
                    in_tab = False
                    continue
                if in_tab:
                    if r'\toprule' in tl or r'\midrule' in tl or r'\bottomrule' in tl:
                        continue
                    if '&' in tl:
                        cells = [c.strip().replace(r'\\', '') for c in tl.split('&')]
                        row_cells = []
                        for c in cells:
                            txt = f"<b>{c}</b>" if len(rows) == 0 else c
                            p_cell = safe_paragraph(txt, styles['TableFoot'])
                            row_cells.append(p_cell if p_cell else safe_paragraph("-", styles['TableFoot']))
                        rows.append(row_cells)
            
            if rows:
                col_count = len(rows[0])
                col_w = 523 / col_count
                t = Table(rows, colWidths=[col_w]*col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ]))
                story.append(t)
                story.append(Spacer(1, 4))
            continue

        # Handle itemize / enumerate
        if line.startswith(r'\item'):
            item_text = line.replace(r'\item', '').strip()
            p = safe_paragraph(f"• {item_text}", styles['Bullet'])
            if p: story.append(p)
            continue

        # Handle Equations
        if line.startswith(r'\begin{equation}'):
            eq_lines = []
            while i < len(lines) and not lines[i].strip().startswith(r'\end{equation}'):
                eq_lines.append(lines[i].strip())
                i += 1
            i += 1
            eq_str = " ".join(eq_lines)
            p = safe_paragraph(f"<b>[Equation]</b> <i>{eq_str}</i>", styles['Equation'])
            if p: story.append(p)
            continue

        # Handle regular paragraphs
        if line and not line.startswith('%') and not line.startswith('\\') and len(line) > 15:
            p = safe_paragraph(line, styles['Body'])
            if p: story.append(p)

    # Back matter: CRediT, Declarations, References
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1E3A8A'), spaceBefore=4, spaceAfter=8))
    story.append(safe_paragraph("<b>References</b>", styles['H1']))
    
    # Read references
    ref_entries = []
    in_bib = False
    cur_ref = ""
    for l in lines:
        if r'\begin{thebibliography}' in l:
            in_bib = True
            continue
        if r'\end{thebibliography}' in l:
            in_bib = False
            if cur_ref:
                ref_entries.append(cur_ref)
            continue
        if in_bib:
            if l.strip().startswith(r'\bibitem'):
                if cur_ref:
                    ref_entries.append(cur_ref)
                m = re.search(r'\\bibitem\{([^}]+)\}', l)
                cur_ref = f"[{len(ref_entries)+1}] "
            else:
                if l.strip() and not l.strip().startswith('%') and not l.strip().startswith('\\'):
                    cur_ref += l.strip() + " "
    
    for r in ref_entries:
        p = safe_paragraph(r.strip(), styles['TableFoot'])
        if p: story.append(p)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")

if __name__ == '__main__':
    build_pdf()
