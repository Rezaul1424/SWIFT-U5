"""
Script to compile docs/phase6_final_reconciliation_report.md into a publication-quality PDF using ReportLab.
"""
import re
import os
import sys
from pathlib import Path
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
            self.drawString(36, 812, "SWIFT-U5 | Final Phase 6 Reconciliation & Results Freeze Report")
            self.drawRightString(559, 812, "Methodological Verification Deliverable")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.5)
            self.line(36, 806, 559, 806)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(36, 36, 559, 36)
        
        self.drawString(36, 25, "SWIFT-U5 Study • BDHS 2004–2022 • Final Benchmark Results Freeze")
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
    inner = inner.replace(r"\rightarrow", "→")
    inner = inner.replace(r"\in", "∈")
    inner = inner.replace(r"\Delta", "Δ")
    inner = inner.replace(r"\bar{\pi}_w", "π̄_w")
    inner = inner.replace(r"\bar{\pi}", "π̄")
    inner = inner.replace(r"\hat{p}", "p̂")
    inner = inner.replace(r"\tau", "τ")
    inner = inner.replace(r"\le", "≤")
    inner = inner.replace(r"\ge", "≥")
    inner = inner.replace(r"\text{AUROC}", "AUROC")
    inner = inner.replace(r"\text{AUPRC}", "AUPRC")
    inner = inner.replace(r"\text{Brier}", "Brier")
    inner = inner.replace(r"\text{NNS}", "NNS")
    inner = inner.replace(r"\text{Sens}", "Sensitivity")
    inner = inner.replace(r"\text{NB}", "NB")
    inner = inner.replace(r"\text{All}", "All")
    inner = inner.replace(r"\text{None}", "None")
    inner = inner.replace(r"\text{weight}", "weight")
    inner = inner.replace(r"\text{val}", "val")
    inner = inner.replace(r"\text{Arm A}", "Arm A")
    inner = inner.replace(r"\text{Arm B}", "Arm B")
    inner = inner.replace(r"\text{Arm C}", "Arm C")
    inner = inner.replace(r"\text{Arm D}", "Arm D")
    inner = inner.replace(r"\text{Inflation}", "Inflation")
    inner = inner.replace(r"\text{Interaction}", "Interaction")
    inner = inner.replace(r"\text{Output}", "Output")
    inner = inner.replace(r"\text{Calculated Output}", "Calculated Output")
    inner = inner.replace(r"\text{Unweighted}", "Unweighted")
    inner = inner.replace(r"\text{Weighted}", "Weighted")
    inner = inner.replace(r"\text{hist}", "hist")
    inner = re.sub(r'\\mathbf\{([^}]+)\}', r'<b>\1</b>', inner)
    inner = inner.replace(r"\mathbf", "")
    inner = inner.replace(r"\frac", "")
    inner = inner.replace(r"\left(", "(").replace(r"\right)", ")")
    inner = inner.replace(r"\{", "{").replace(r"\}", "}")
    inner = re.sub(r'\{([^{}]+)\}\{([^{}]+)\}', r'(\1 / \2)', inner)
    inner = inner.replace("{", "").replace("}", "")
    return inner

def clean_inline_formatting(text):
    text = re.sub(r'\$\$([^$]+)\$\$', lambda m: f"<b>{convert_math(m.group(0))}</b>", text)
    text = re.sub(r'\$([^$]+)\$', lambda m: convert_math(m.group(0)), text)
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" color="#0F172A"><b>\1</b></font>', text)
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
        # Table 3: Model, Cut, Test Round, Val Eff, Wt Eff, Inter, Inf, NNS
        col_widths = [65, 80, 45, 65, 65, 65, 65, 73]
    elif num_cols == 9:
        if 'Target' in headers[1] or 'Training' in headers[2]:
            # Calibration provenance table
            col_widths = [45, 45, 65, 65, 80, 50, 60, 45, 68]
        else:
            # DCA Table
            col_widths = [38, 48, 48, 45, 52, 52, 48, 48, 149]
    else:
        w = avail_width / num_cols
        col_widths = [w] * num_cols

    cell_style = ParagraphStyle(
        'TableCell',
        fontName='ArialUnicode',
        fontSize=5.5 if num_cols >= 8 else 6.2,
        leading=7.2 if num_cols >= 8 else 8.2,
        textColor=colors.HexColor('#1E293B')
    )
    cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Arial-Bold',
        fontSize=5.5 if num_cols >= 8 else 6.2,
        leading=7.2 if num_cols >= 8 else 8.2,
        textColor=colors.HexColor('#0F172A')
    )
    header_style = ParagraphStyle(
        'TableHeader',
        fontName='Arial-Bold',
        fontSize=5.5 if num_cols >= 8 else 6.2,
        leading=7.2 if num_cols >= 8 else 8.2,
        textColor=colors.white
    )

    table_data = []
    # Header row
    hdr_row = []
    for h in headers:
        clean_h = clean_inline_formatting(h)
        hdr_row.append(Paragraph(clean_h, header_style))
    table_data.append(hdr_row)

    # Data rows
    for r_idx, dline in enumerate(data_lines):
        raw_cells = [c.strip() for c in dline.split('|')[1:-1]]
        if len(raw_cells) < num_cols:
            raw_cells += [''] * (num_cols - len(raw_cells))
        row = []
        is_highlight = False
        for c in raw_cells:
            clean_c = clean_inline_formatting(c)
            if '<b>' in clean_c or 'PRIMARY' in clean_c or 'Frozen' in clean_c or 'Total' in clean_c:
                row.append(Paragraph(clean_c, cell_bold))
            else:
                row.append(Paragraph(clean_c, cell_style))
        table_data.append(row)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    ts = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
        ('LEFTPADDING', (0, 0), (-1, -1), 2.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CBD5E1')),
    ]
    # Alternating row colors
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            ts.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F8FAFC')))
    t.setStyle(TableStyle(ts))
    return t

def build_pdf(md_path: Path, output_pdf_path: Path):
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44
    )

    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    story = []

    in_table = False
    table_lines = []
    in_code = False
    code_lines = []
    in_callout = False
    callout_lines = []

    for line in lines:
        stripped = line.strip()

        # Handle Code Blocks
        if stripped.startswith('```'):
            if in_code:
                in_code = False
                code_text = "<br/>".join([clean_inline_formatting(cl).replace(' ', '&nbsp;') for cl in code_lines])
                code_p = Paragraph(code_text, base_styles['Code'])
                t = Table([[code_p]], colWidths=[523])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 4))
                code_lines = []
            else:
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        # Handle Callout Blocks
        if stripped.startswith('>'):
            in_callout = True
            c_text = stripped.lstrip('>').strip()
            if c_text.startswith('[!NOTE]') or c_text.startswith('[!IMPORTANT]') or c_text.startswith('[!WARNING]'):
                c_text = f"<b>{c_text}</b>"
            callout_lines.append(clean_inline_formatting(c_text))
            continue
        elif in_callout:
            in_callout = False
            cp = Paragraph("<br/>".join(callout_lines), base_styles['CalloutText'])
            t = Table([[cp]], colWidths=[523])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EFF6FF')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#93C5FD')),
                ('LINEBEFORE', (0, 0), (0, -1), 2.5, colors.HexColor('#2563EB')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 4))
            callout_lines = []

        # Handle Markdown Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            in_table = False
            tbl_flow = process_table(table_lines)
            if tbl_flow:
                story.append(tbl_flow)
                story.append(Spacer(1, 5))
            table_lines = []

        if not stripped:
            continue

        # Horizontal rules
        if stripped == '---':
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceBefore=3, spaceAfter=4))
            continue

        # Headings
        if stripped.startswith('# '):
            story.append(Paragraph(clean_inline_formatting(stripped[2:]), base_styles['Title']))
        elif stripped.startswith('## '):
            story.append(Paragraph(clean_inline_formatting(stripped[3:]), base_styles['H2']))
        elif stripped.startswith('### '):
            story.append(Paragraph(clean_inline_formatting(stripped[4:]), base_styles['H3']))
        elif stripped.startswith('#### '):
            story.append(Paragraph(clean_inline_formatting(stripped[5:]), base_styles['H4']))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = f"• {clean_inline_formatting(stripped[2:])}"
            story.append(Paragraph(bullet_text, base_styles['Bullet']))
        elif re.match(r'^\d+\.\s', stripped):
            num_match = re.match(r'^(\d+\.)\s(.*)', stripped)
            num_prefix = num_match.group(1)
            num_text = f"<b>{num_prefix}</b> {clean_inline_formatting(num_match.group(2))}"
            story.append(Paragraph(num_text, base_styles['Bullet']))
        elif stripped.startswith('$$') and stripped.endswith('$$'):
            story.append(Paragraph(clean_inline_formatting(stripped), base_styles['DisplayMath']))
        else:
            story.append(Paragraph(clean_inline_formatting(stripped), base_styles['Body']))

    # Cleanup trailing table if file ends with table
    if in_table and table_lines:
        tbl_flow = process_table(table_lines)
        if tbl_flow:
            story.append(tbl_flow)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF: {output_pdf_path} ({output_pdf_path.stat().st_size / 1024:.1f} KB)")

if __name__ == '__main__':
    md_file = Path('docs/phase6_final_reconciliation_report.md')
    pdf_out1 = Path('docs/phase6_final_reconciliation_report.pdf')
    pdf_out2 = Path('/Users/rezaul1424/.gemini/antigravity/brain/0addd384-01e5-47af-9194-9a791153f016/phase6_final_reconciliation_report.pdf')
    
    build_pdf(md_file, pdf_out1)
    build_pdf(md_file, pdf_out2)
