"""
Script to compile phase6_execution_report.md into a professional publication-quality PDF using ReportLab.
"""
import re
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# 1. Register fonts
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
            self.drawString(36, 812, "SWIFT-U5 | Phase 6 Results Reconciliation & Verification Report")
            self.drawRightString(559, 812, "Prospective Benchmark Deliverable")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.5)
            self.line(36, 806, 559, 806)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(36, 36, 559, 36)
        
        self.drawString(36, 25, "SWIFT-U5 Project • Reproducibility & Research Design Protocol • BDHS 2004–2022")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 25, page_text)
        self.restoreState()

# Base Styles
base_styles = {
    'Title': ParagraphStyle('ReportTitle', fontName='Arial-Bold', fontSize=13, leading=16.5, textColor=colors.HexColor('#0F294A'), spaceAfter=3),
    'Subtitle': ParagraphStyle('ReportSubtitle', fontName='ArialUnicode', fontSize=7.6, leading=10, textColor=colors.HexColor('#475569'), spaceAfter=5),
    'H2': ParagraphStyle('Heading2', fontName='Arial-Bold', fontSize=9.5, leading=12.5, textColor=colors.HexColor('#1E3A8A'), spaceBefore=6, spaceAfter=2, keepWithNext=True),
    'H3': ParagraphStyle('Heading3', fontName='Arial-Bold', fontSize=8.2, leading=11, textColor=colors.HexColor('#2563EB'), spaceBefore=4, spaceAfter=1.5, keepWithNext=True),
    'H4': ParagraphStyle('Heading4', fontName='Arial-Bold', fontSize=7.6, leading=9.8, textColor=colors.HexColor('#334155'), spaceBefore=3, spaceAfter=1, keepWithNext=True),
    'Body': ParagraphStyle('BodyTextCustom', fontName='ArialUnicode', fontSize=7.0, leading=9.8, textColor=colors.HexColor('#1E293B'), spaceAfter=2),
    'Bullet': ParagraphStyle('BulletCustom', fontName='ArialUnicode', fontSize=7.0, leading=9.8, textColor=colors.HexColor('#1E293B'), leftIndent=10, spaceAfter=1.5),
    'Code': ParagraphStyle('CodeBlock', fontName='Courier', fontSize=6.5, leading=8.2, textColor=colors.HexColor('#0F172A')),
    'CalloutText': ParagraphStyle('CalloutT', fontName='ArialUnicode', fontSize=7.0, leading=9.8, textColor=colors.HexColor('#1E3A8A')),
    'DisplayMath': ParagraphStyle('DisplayM', fontName='ArialUnicode', fontSize=7.5, leading=10, textColor=colors.HexColor('#0F294A'), alignment=1, spaceBefore=2, spaceAfter=2),
}

def convert_math(m_str):
    """Convert LaTeX math string to clean HTML entities for ReportLab."""
    if m_str.startswith("$$") and m_str.endswith("$$"):
        inner = m_str[2:-2].strip()
    elif m_str.startswith("$") and m_str.endswith("$"):
        inner = m_str[1:-1].strip()
    else:
        return m_str

    inner = inner.replace(r"\%", "%")
    inner = inner.replace(r"\times", "×")
    inner = inner.replace(r"\approx", "≈")
    inner = inner.replace(r"\cdot", "·")
    inner = inner.replace(r"\to", "→")
    inner = inner.replace(r"\in", "∈")
    inner = inner.replace(r"\Delta", "Δ")
    inner = inner.replace(r"\bar{\pi}_w", "π̄_w")
    inner = inner.replace(r"\bar{\pi}", "π̄")
    inner = inner.replace(r"\hat{p}", "p̂")
    inner = inner.replace(r"\bar{w}", "w̄")
    inner = inner.replace(r"\tau", "τ")
    inner = inner.replace(r"\le", "≤")
    inner = inner.replace(r"\ge", "≥")
    inner = inner.replace(r"\text{AUROC}_W", "AUROC_W")
    inner = inner.replace(r"\text{AUROC}", "AUROC")
    inner = inner.replace(r"\text{AUPRC}", "AUPRC")
    inner = inner.replace(r"\text{Brier}", "Brier")
    inner = inner.replace(r"\text{Slope}", "Slope")
    inner = inner.replace(r"\text{Intercept}", "Intercept")
    inner = inner.replace(r"\text{BSS}", "BSS")
    inner = inner.replace(r"\text{NNS}_{5\%}", "NNS_5%")
    inner = inner.replace(r"\text{NNS}_{10\%}", "NNS_10%")
    inner = inner.replace(r"\text{NNS}", "NNS")
    inner = inner.replace(r"\text{Sens}", "Sensitivity")
    inner = inner.replace(r"\text{Development }", "Development ")
    inner = inner.replace(r"\text{Validation }", "Validation ")
    inner = inner.replace(r"\text{Locked Test }", "Locked Test ")
    inner = inner.replace(r"\text{model}", "model")
    inner = inner.replace(r"\text{null}", "null")
    inner = inner.replace(r"\text{XGBoost}", "XGBoost")
    inner = inner.replace(r"\text{ skill over null}", " skill over null")
    inner = inner.replace(r"\text{catastrophic miscalibration}", "catastrophic miscalibration")
    inner = re.sub(r'\\mathbf\{([^}]+)\}', r'<b>\1</b>', inner)
    inner = inner.replace(r"\mathbf", "")
    inner = inner.replace(r"\frac", "")
    inner = inner.replace(r"\left(", "(").replace(r"\right)", ")")
    inner = inner.replace(r"\{", "{").replace(r"\}", "}")
    
    # Handle frac braces if present
    inner = re.sub(r'\{([^{}]+)\}\{([^{}]+)\}', r'(\1 / \2)', inner)
    
    # KaTeX KaTeX symbols cleanup
    inner = inner.replace("{", "").replace("}", "")
    return inner

def clean_inline_formatting(text):
    """Replace Markdown bold/italic/code/math with ReportLab HTML tags."""
    # Convert math
    text = re.sub(r'\$\$([^$]+)\$\$', lambda m: f"<b>{convert_math(m.group(0))}</b>", text)
    text = re.sub(r'\$([^$]+)\$', lambda m: convert_math(m.group(0)), text)
    
    # Convert bold+italic
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'<b><i>\1</i></b>', text)
    # Convert bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Convert italic
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    # Convert inline code
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" color="#0F172A"><b>\1</b></font>', text)
    # Convert markdown links [text](url) -> <u>text</u>
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'<u>\1</u>', text)
    return text

def process_table(table_lines):
    if len(table_lines) < 2:
        return None
    
    header_line = table_lines[0]
    align_line = table_lines[1]
    data_lines = table_lines[2:]

    headers = [c.strip() for c in header_line.split('|')[1:-1]]
    num_cols = len(headers)

    col_widths = []
    avail_width = 523
    
    # Dynamic column widths based on column count
    if num_cols == 2:
        col_widths = [160, 363]
    elif num_cols == 3:
        col_widths = [140, 180, 203]
    elif num_cols == 4:
        col_widths = [110, 110, 140, 163]
    elif num_cols == 5:
        col_widths = [110, 95, 95, 110, 113]
    elif num_cols == 6:
        col_widths = [90, 85, 85, 85, 85, 93]
    elif num_cols == 7:
        col_widths = [75, 75, 75, 75, 75, 75, 73]
    elif num_cols == 8:
        col_widths = [85, 60, 50, 60, 75, 65, 65, 63]
    elif num_cols == 9:
        col_widths = [90, 45, 45, 55, 68, 62, 52, 52, 54]
    elif num_cols == 10:
        col_widths = [85, 40, 40, 48, 60, 52, 52, 48, 48, 50]
    elif num_cols == 11:
        col_widths = [80, 38, 38, 45, 58, 48, 48, 42, 42, 42, 42]
    else:
        w = avail_width / num_cols
        col_widths = [w] * num_cols

    pad_tb = 2.0 if num_cols >= 8 else 3.0
    pad_lr = 2.0 if num_cols >= 8 else 3.5
    f_size = 5.6 if num_cols >= 10 else (6.2 if num_cols >= 8 else 6.8)
    l_size = 7.2 if num_cols >= 10 else (8.0 if num_cols >= 8 else 8.8)

    th_style = ParagraphStyle('TH', fontName='Arial-Bold', fontSize=f_size, leading=l_size, textColor=colors.white)
    td_style = ParagraphStyle('TD', fontName='ArialUnicode', fontSize=f_size, leading=l_size, textColor=colors.HexColor('#1E293B'))

    table_data = []
    # Header row
    hdr_data = [Paragraph(clean_inline_formatting(h), th_style) for h in headers]
    table_data.append(hdr_data)

    for row in data_lines:
        cells = [c.strip() for c in row.split('|')[1:-1]]
        if len(cells) < num_cols:
            cells += [''] * (num_cols - len(cells))
        elif len(cells) > num_cols:
            cells = cells[:num_cols]

        row_data = []
        for cell in cells:
            cleaned = clean_inline_formatting(cell)
            row_data.append(Paragraph(cleaned, td_style))
        table_data.append(row_data)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,0), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), pad_tb),
        ('BOTTOMPADDING', (0,0), (-1,-1), pad_tb),
        ('LEFTPADDING', (0,0), (-1,-1), pad_lr),
        ('RIGHTPADDING', (0,0), (-1,-1), pad_lr),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]
    for r_idx in range(1, len(table_data)):
        bg = colors.HexColor('#F8FAFC') if r_idx % 2 == 1 else colors.white
        t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))

    t.setStyle(TableStyle(t_style))
    return t

def parse_markdown_to_flowables(md_content):
    lines = md_content.split('\n')
    flowables = []
    i = 0
    n = len(lines)

    in_code_block = False
    code_lines = []
    table_lines = []

    while i < n:
        line = lines[i]

        # 1. Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                code_text = "<br/>".join([l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for l in code_lines])
                p = Paragraph(code_text, base_styles['Code'])
                t = Table([[p]], colWidths=[523])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 3),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ]))
                flowables.append(Spacer(1, 2))
                flowables.append(t)
                flowables.append(Spacer(1, 2.5))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # 2. Markdown tables
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            table_lines.append(line.strip())
            i += 1
            while i < n and lines[i].strip().startswith('|') and '|' in lines[i].strip()[1:]:
                table_lines.append(lines[i].strip())
                i += 1
            
            parsed_table = process_table(table_lines)
            if parsed_table:
                flowables.append(Spacer(1, 2))
                flowables.append(parsed_table)
                flowables.append(Spacer(1, 2.5))
            table_lines = []
            continue

        # 3. Callout blocks > [!NOTE] or > [!IMPORTANT]
        if line.strip().startswith('>'):
            callout_lines = [line.strip()[1:].strip()]
            i += 1
            while i < n and lines[i].strip().startswith('>'):
                callout_lines.append(lines[i].strip()[1:].strip())
                i += 1
            
            full_callout = " ".join(callout_lines)
            is_important = '[!IMPORTANT]' in full_callout or 'IMPORTANT' in full_callout
            bg_color = colors.HexColor('#FFFBEB') if is_important else colors.HexColor('#EFF6FF')
            border_color = colors.HexColor('#F59E0B') if is_important else colors.HexColor('#3B82F6')
            badge_title = "CRITICAL PROTOCOL VERDICT" if is_important else "NOTE / IMPLEMENTATION CONTEXT"
            badge_color = "#B45309" if is_important else "#1D4ED8"
            
            cleaned = full_callout.replace('[!NOTE]', '').replace('[!IMPORTANT]', '').strip()
            cleaned = clean_inline_formatting(cleaned)
            content_html = f"<b><font color='{badge_color}'>{badge_title}:</font></b> {cleaned}"
            
            cp = Paragraph(content_html, base_styles['CalloutText'])
            ct = Table([[cp]], colWidths=[523])
            ct.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), bg_color),
                ('LINEBEFORE', (0,0), (0,-1), 3.0, border_color),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            flowables.append(Spacer(1, 2))
            flowables.append(ct)
            flowables.append(Spacer(1, 2.5))
            continue

        # 4. Display Math $$ ... $$
        if line.strip().startswith('$$') and line.strip().endswith('$$') and len(line.strip()) > 4:
            math_content = convert_math(line.strip())
            p = Paragraph(math_content, base_styles['DisplayMath'])
            t = Table([[p]], colWidths=[523])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            flowables.append(Spacer(1, 1.5))
            flowables.append(t)
            flowables.append(Spacer(1, 2))
            i += 1
            continue

        # 5. Horizontal Rule ---
        if line.strip() in ['---', '***', '___']:
            flowables.append(Spacer(1, 1.5))
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceBefore=1.5, spaceAfter=2.5))
            i += 1
            continue

        # 6. Headings
        if line.startswith('# '):
            title_text = clean_inline_formatting(line[2:].strip())
            flowables.append(Paragraph(title_text, base_styles['Title']))
            i += 1
            continue
        elif line.startswith('## '):
            h2_text = clean_inline_formatting(line[3:].strip())
            flowables.append(Spacer(1, 3.5))
            flowables.append(Paragraph(h2_text, base_styles['H2']))
            flowables.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#CBD5E1'), spaceBefore=1, spaceAfter=2.5))
            i += 1
            continue
        elif line.startswith('### '):
            h3_text = clean_inline_formatting(line[4:].strip())
            flowables.append(Spacer(1, 2.5))
            flowables.append(Paragraph(h3_text, base_styles['H3']))
            i += 1
            continue
        elif line.startswith('#### '):
            h4_text = clean_inline_formatting(line[5:].strip())
            flowables.append(Spacer(1, 2))
            flowables.append(Paragraph(h4_text, base_styles['H4']))
            i += 1
            continue

        # 7. Unordered / Ordered Lists
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped):
            bullet_match = re.match(r'^(\d+\.|\-|\*)\s+(.*)$', stripped)
            if bullet_match:
                prefix = bullet_match.group(1)
                text = bullet_match.group(2)
                bullet_prefix = "&bull; " if prefix in ['-', '*'] else f"<b>{prefix}</b> "
                cleaned = clean_inline_formatting(text)
                flowables.append(Paragraph(f"{bullet_prefix}{cleaned}", base_styles['Bullet']))
            i += 1
            continue

        # 8. Regular text paragraph
        if stripped:
            cleaned = clean_inline_formatting(stripped)
            flowables.append(Paragraph(cleaned, base_styles['Body']))

        i += 1

    return flowables

def build_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    flowables = parse_markdown_to_flowables(md_content)
    doc.build(flowables, canvasmaker=NumberedCanvas)
    print(f"Successfully generated clean PDF: {pdf_path}")

if __name__ == '__main__':
    md_file = '/Users/rezaul1424/Everything/SWIFT-U5/docs/phase6_execution_report.md'
    pdf_brain = '/Users/rezaul1424/.gemini/antigravity/brain/0addd384-01e5-47af-9194-9a791153f016/phase6_execution_report.pdf'
    pdf_docs = '/Users/rezaul1424/Everything/SWIFT-U5/docs/phase6_execution_report.pdf'
    
    build_pdf(md_file, pdf_brain)
    build_pdf(md_file, pdf_docs)
