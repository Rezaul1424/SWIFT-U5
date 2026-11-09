"""
Script to compile probability_and_weights_diagnostic.md into a professional publication-quality PDF using ReportLab.
"""
import re
import os
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
            self.drawString(36, 812, "SWIFT-U5 | Probability Calibration & Weight Dominance Diagnostic")
            self.drawRightString(559, 812, "Methodological Audit & Sensitivity Analysis")
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
    'Title': ParagraphStyle('ReportTitle', fontName='Arial-Bold', fontSize=14, leading=17.5, textColor=colors.HexColor('#0F294A'), spaceAfter=3),
    'Subtitle': ParagraphStyle('ReportSubtitle', fontName='ArialUnicode', fontSize=7.8, leading=10.5, textColor=colors.HexColor('#475569'), spaceAfter=5),
    'H2': ParagraphStyle('Heading2', fontName='Arial-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#1E3A8A'), spaceBefore=7, spaceAfter=2, keepWithNext=True),
    'H3': ParagraphStyle('Heading3', fontName='Arial-Bold', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#2563EB'), spaceBefore=4.5, spaceAfter=1.5, keepWithNext=True),
    'H4': ParagraphStyle('Heading4', fontName='Arial-Bold', fontSize=7.8, leading=10, textColor=colors.HexColor('#334155'), spaceBefore=3.5, spaceAfter=1, keepWithNext=True),
    'Body': ParagraphStyle('BodyTextCustom', fontName='ArialUnicode', fontSize=7.2, leading=10, textColor=colors.HexColor('#1E293B'), spaceAfter=2),
    'Bullet': ParagraphStyle('BulletCustom', fontName='ArialUnicode', fontSize=7.2, leading=10, textColor=colors.HexColor('#1E293B'), leftIndent=10, spaceAfter=1.5),
    'Code': ParagraphStyle('CodeBlock', fontName='Courier', fontSize=6.8, leading=8.5, textColor=colors.HexColor('#0F172A')),
    'CalloutText': ParagraphStyle('CalloutT', fontName='ArialUnicode', fontSize=7.2, leading=10, textColor=colors.HexColor('#1E3A8A')),
    'DisplayMath': ParagraphStyle('DisplayM', fontName='ArialUnicode', fontSize=7.8, leading=10.5, textColor=colors.HexColor('#0F294A'), alignment=1, spaceBefore=2, spaceAfter=2),
}

def convert_math(m_str):
    """Convert LaTeX math string to clean HTML entities for ReportLab."""
    if m_str.startswith("$$") and m_str.endswith("$$"):
        inner = m_str[2:-2].strip()
        is_display = True
    elif m_str.startswith("$") and m_str.endswith("$"):
        inner = m_str[1:-1].strip()
        is_display = False
    else:
        return m_str

    inner = inner.replace(r"\%", "%")
    inner = inner.replace(r"\times", "×")
    inner = inner.replace(r"\approx", "≈")
    inner = inner.replace(r"\cdot", "·")
    inner = inner.replace(r"\to", "→")
    inner = inner.replace(r"\in", "∈")
    inner = inner.replace(r"\Delta t", "Δt")
    inner = inner.replace(r"\Delta", "Δ")
    inner = inner.replace(r"\exp", "exp")
    inner = inner.replace(r"\sigma", "σ")
    inner = inner.replace(r"\left(", "(").replace(r"\right)", ")")
    inner = inner.replace(r"\quad \text{where }", " &nbsp;&nbsp;&nbsp;&nbsp; <b>where</b> ")
    inner = inner.replace(r"\quad \text{vs} \quad", " &nbsp;&nbsp; <b>vs</b> &nbsp;&nbsp; ")
    inner = inner.replace(r"\quad", " &nbsp;&nbsp; ")
    inner = inner.replace(r"\;", " ")
    inner = inner.replace(r"\text{ difference}", " difference")
    inner = inner.replace(r"\text{dev}", "dev")
    inner = inner.replace(r"\text{Target}", "Target")
    inner = inner.replace(r"\text{Source}", "Source")
    inner = inner.replace(r"\text{target}", "target")
    inner = inner.replace(r"\text{source}", "source")
    inner = inner.replace(r"\text{AUROC}", "AUROC")
    inner = inner.replace(r"\text{AUPRC}", "AUPRC")
    inner = inner.replace(r"\text{UW}", "UW")
    inner = inner.replace(r"\text{Mean}", "Mean")
    inner = inner.replace(r"\text{ObsPrev}", "ObsPrev")
    inner = inner.replace(r"\text{surv}", "surv")
    inner = inner.replace(r"\text{time}", "time")
    inner = inner.replace(r"\text{shift}", "shift")
    inner = inner.replace(r"\text{std}", "std")
    inner = inner.replace(r"\text{final}", "final")
    inner = inner.replace(r"\max", "max")
    inner = inner.replace(r"a^*", "<i>a</i>*").replace(r"b^*", "<i>b</i>*")
    inner = inner.replace(r"\tau_W", "τ<sub>W</sub>")
    inner = inner.replace(r"\tau_{UW}", "τ<sub>UW</sub>")
    inner = inner.replace(r"\tau", "τ")
    inner = inner.replace(r"\bar{W}_{2014}^2", "W̄<sub>2014</sub>²")
    inner = inner.replace(r"\bar{W}_{2014}", "W̄<sub>2014</sub>")
    inner = inner.replace(r"\bar{W}_{2004}", "W̄<sub>2004</sub>")
    inner = inner.replace(r"\bar{W}_k", "W̄<sub>k</sub>")
    inner = inner.replace(r"\bar{w}_{surv}", "w̄<sub>surv</sub>")
    inner = inner.replace(r"\bar{w}_{time}", "w̄<sub>time</sub>")
    inner = inner.replace(r"\bar{w}_{shift}", "w̄<sub>shift</sub>")
    inner = inner.replace(r"\frac{(29,686)^2}{165,754}", "(29,686)² / 165,754")
    inner = inner.replace(r"\frac{\sum W_k^2}{\sum W^2}", "ΣW<sub>k</sub>² / ΣW²")
    inner = inner.replace(r"\frac{\sum W_k}{\sum W}", "ΣW<sub>k</sub> / ΣW")
    inner = inner.replace(r"\frac{Mean}{ObsPrev}", "Mean / ObsPrev")
    inner = inner.replace(r"\frac{\text{Mean}}{\text{ObsPrev}}", "Mean / ObsPrev")
    inner = inner.replace(r"\sum W_i^2", "ΣW<sub>i</sub>²")
    inner = inner.replace(r"\sum W_i", "ΣW<sub>i</sub>")
    inner = inner.replace(r"\sum", "Σ")
    inner = inner.replace(r"\ln(19.3)", "ln(19.3)")
    inner = inner.replace(r"^2", "²")
    inner = inner.replace(r"_W", "<sub>W</sub>")
    inner = inner.replace(r"_B", "<sub>B</sub>")
    inner = inner.replace(r"_i", "<sub>i</sub>")
    inner = inner.replace(r"_k", "<sub>k</sub>")
    inner = inner.replace(r"_{\text{dev}}", "<sub>dev</sub>")
    inner = inner.replace(r"_{dev}", "<sub>dev</sub>")
    inner = inner.replace(r"_{std}", "<sub>std</sub>")
    inner = inner.replace(r"_{\max}", "<sub>max</sub>")
    inner = inner.replace(r"_{max}", "<sub>max</sub>")
    inner = inner.replace(r"_{final}", "<sub>final</sub>")
    inner = inner.replace(r"_{\text{UW}}", "<sub>UW</sub>")
    inner = inner.replace(r"\mathbf{0.048557}", "<b>0.048557</b>")
    inner = inner.replace(r"\mathbf{92.7%}", "<b>92.7%</b>")
    inner = inner.replace(r"\mathbf{5,316.7}", "<b>5,316.7</b>")
    inner = inner.replace(r"N_+", "N<sub>+</sub>")

    if is_display:
        return f'<font color="#0F294A"><b>{inner}</b></font>'
    else:
        return inner

def clean_inline_formatting(text):
    """Convert markdown inline formatting to reportlab html tags."""
    # Badges
    text = text.replace('🔄 PENDING', '<font color="#B45309"><b>[PENDING]</b></font>')
    text = text.replace('🔄', '<font color="#B45309"><b>[PENDING]</b></font>')
    text = text.replace('✅ PASS', '<font color="#15803D"><b>[PASS]</b></font>')
    text = text.replace('✅ VERIFIED', '<font color="#15803D"><b>[VERIFIED]</b></font>')
    text = text.replace('✅ COMPLETE', '<font color="#15803D"><b>[COMPLETE]</b></font>')
    text = text.replace('✅ DECISION MADE', '<font color="#15803D"><b>[DECISION MADE]</b></font>')
    text = text.replace('✅ READY', '<font color="#15803D"><b>[READY]</b></font>')
    text = text.replace('✅', '<font color="#15803D"><b>✔</b></font>')

    # Convert standalone ampersands not already part of an HTML entity
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|nbsp);)", "&amp;", text)

    # Convert math
    text = re.sub(r"\$\$[^\$]+\$\$|\$[^\$\n]+\$", lambda m: convert_math(m.group(0)), text)

    # Convert backticks to Courier font
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" color="#1E3A8A">\1</font>', text)

    # Bold **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    # Italic *text*
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)

    # Links [text](url) -> <u>text</u>
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"<u>\1</u>", text)

    return text

def parse_markdown_to_flowables(md_content):
    flowables = []
    lines = md_content.split('\n')
    i = 0
    n = len(lines)

    in_code_block = False
    code_lines = []
    table_lines = []

    while i < n:
        line = lines[i]

        # 1. Code blocks ```
        if line.strip().startswith('```'):
            if in_code_block:
                code_text = "<br/>".join(
                    line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    for line in code_lines
                )
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
            flowables.append(Paragraph(h3_text, base_styles['H3']))
            i += 1
            continue
        elif line.startswith('#### '):
            h4_text = clean_inline_formatting(line[5:].strip())
            flowables.append(Paragraph(h4_text, base_styles['H4']))
            i += 1
            continue

        # 7. Bullets (- or *)
        if line.strip().startswith(('- ', '* ')):
            bullet_text = clean_inline_formatting(line.strip()[2:].strip())
            flowables.append(Paragraph(f"• &nbsp;{bullet_text}", base_styles['Bullet']))
            i += 1
            continue

        # 8. Numbered list
        m_num = re.match(r'^(\d+)\.\s+(.*)$', line.strip())
        if m_num:
            num = m_num.group(1)
            num_text = clean_inline_formatting(m_num.group(2).strip())
            flowables.append(Paragraph(f"<b>{num}.</b> &nbsp;{num_text}", base_styles['Bullet']))
            i += 1
            continue

        # 9. Metadata / Subtitle line right under title
        if line.strip().startswith('**Date:**'):
            meta_text = clean_inline_formatting(line.strip())
            flowables.append(Paragraph(meta_text, base_styles['Subtitle']))
            i += 1
            continue

        # 10. Standard paragraph
        text = line.strip()
        if text:
            cleaned_p = clean_inline_formatting(text)
            flowables.append(Paragraph(cleaned_p, base_styles['Body']))
        else:
            flowables.append(Spacer(1, 1.5))
        
        i += 1

    return flowables

def process_table(raw_lines):
    rows = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split('|')]
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        
        if all(re.match(r'^:?-+:?$', c) for c in cells if c):
            continue
        
        rows.append(cells)

    if not rows:
        return None

    num_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < num_cols:
            r.append('')

    TOTAL_WIDTH = 523.0

    # Determine column widths and font sizes based on table column count
    if num_cols == 13:
        first_header = rows[0][0].lower()
        if 'processing' in first_header or 'stage' in first_header:
            # Table 1A: Stage, Mean, Median, Min, Max, p1, p5, p25, p50, p75, p95, p99, Ratio
            col_widths = [139.0] + [31.0]*11 + [43.0]
        else:
            # Table 3: Cut, Waves, Test, Derivation, tauW, tauUW, ValPrev, TestPrev, MeanProb, Slope, Int, Brier, Transf
            raw = [20.0, 62.0, 32.0, 116.0, 36.0, 36.0, 31.0, 31.0, 33.0, 27.0, 32.0, 28.0, 39.0]
            scale = TOTAL_WIDTH / sum(raw)
            col_widths = [w * scale for w in raw]
        th_font_size = 5.5
        th_leading = 6.8
        td_font_size = 5.5
        td_leading = 6.8
        pad_lr = 2
        pad_tb = 1.5
    elif num_cols == 9:
        first_header = rows[0][0].lower()
        if 'wave' in first_header:
            # Table 2A: Wave, Records, w_surv, w_time, w_shift, W_k, % W, % W^2, ESS
            raw = [38.0, 52.0, 56.0, 52.0, 60.0, 62.0, 55.0, 55.0, 63.0]
            scale = TOTAL_WIDTH / sum(raw)
            col_widths = [w * scale for w in raw]
        else:
            # Table 1B: Stage, Brier, Null, BSS, Slope, Int, ECE, Thresh, Source
            raw = [118.0, 42.0, 40.0, 48.0, 38.0, 44.0, 33.0, 56.0, 104.0]
            scale = TOTAL_WIDTH / sum(raw)
            col_widths = [w * scale for w in raw]
        th_font_size = 6.0
        th_leading = 7.5
        td_font_size = 5.8
        td_leading = 7.5
        pad_lr = 3
        pad_tb = 2
    elif num_cols == 5:
        # Table 2B: Metric, Primary, Equal, Change, Implication
        raw = [115.0, 95.0, 95.0, 50.0, 168.0]
        col_widths = raw
        th_font_size = 6.8
        th_leading = 8.5
        td_font_size = 6.5
        td_leading = 8.5
        pad_lr = 3.5
        pad_tb = 2.5
    else:
        # Generic fallback
        col_max_lens = [0] * num_cols
        for r in rows:
            for c_idx, cell in enumerate(r):
                col_max_lens[c_idx] = max(col_max_lens[c_idx], len(cell))
        total_len = sum(col_max_lens)
        if total_len == 0:
            col_widths = [TOTAL_WIDTH / num_cols] * num_cols
        else:
            raw_widths = [max(30.0, (col_len / total_len) * TOTAL_WIDTH) for col_len in col_max_lens]
            scale = TOTAL_WIDTH / sum(raw_widths)
            col_widths = [w * scale for w in raw_widths]
        th_font_size = 7.0
        th_leading = 8.5
        td_font_size = 6.8
        td_leading = 8.5
        pad_lr = 3
        pad_tb = 2

    th_style = ParagraphStyle('DynTH', fontName='Arial-Bold', fontSize=th_font_size, leading=th_leading, textColor=colors.white)
    td_style = ParagraphStyle('DynTD', fontName='ArialUnicode', fontSize=td_font_size, leading=td_leading, textColor=colors.HexColor('#1E293B'))

    table_data = []
    for row_idx, r in enumerate(rows):
        row_data = []
        is_header = (row_idx == 0)
        for col_idx, cell in enumerate(r):
            cleaned = clean_inline_formatting(cell)
            if is_header:
                p = Paragraph(f"<b>{cleaned}</b>", th_style)
            else:
                p = Paragraph(cleaned, td_style)
            row_data.append(p)
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
    md_file = '/Users/rezaul1424/.gemini/antigravity/brain/0addd384-01e5-47af-9194-9a791153f016/probability_and_weights_diagnostic.md'
    pdf_brain = '/Users/rezaul1424/.gemini/antigravity/brain/0addd384-01e5-47af-9194-9a791153f016/probability_and_weights_diagnostic.pdf'
    pdf_docs = '/Users/rezaul1424/Everything/SWIFT-U5/docs/probability_and_weights_diagnostic.pdf'
    
    build_pdf(md_file, pdf_brain)
    build_pdf(md_file, pdf_docs)
