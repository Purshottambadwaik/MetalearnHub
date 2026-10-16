"""
Comprehensive generator for B.Tech Research Methodology Assignment.
Produces DOCX and PDF documents with high academic rigor.
Institution: Government College of Engineering, Nagpur
Department: Computer Science and Engineering
Project: MetaLearnHub (Group 07)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs" / "research_methodology"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from doc_generator_helpers import set_cell_margins, set_cell_background, add_table_borders, NumberedCanvas

import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DOCX_PATH = DOCS_DIR / "Research_Methodology_Project_Application_Assignment_Group07.docx"
PDF_PATH = DOCS_DIR / "Research_Methodology_Project_Application_Assignment_Group07.pdf"

print(f"Target DOCX: {DOCX_PATH}")
print(f"Target PDF:  {PDF_PATH}")
