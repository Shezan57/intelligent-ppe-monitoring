"""
PDF Report Generator Service

Generates professional PDF reports for daily violation summaries.
Uses ReportLab for PDF generation with charts from Matplotlib.
"""

import os
from datetime import date
from typing import List, Dict, Any

from config.settings import settings


class ReportGenerator:
    """
    Generates professional PDF reports from daily violation data.
    
    Report includes:
    - Header with date and summary
    - Violation statistics table
    - Hourly distribution chart
    - Top 5 violation evidence images
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output PDFs
        """
        self.output_dir = output_dir or settings.report_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_daily_report(
        self,
        report_date: date,
        violations: List,
        stats: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive PDF report for a given date.
        
        Args:
            report_date: Date for the report
            violations: List of Violation records
            stats: Pre-calculated statistics dict
            
        Returns:
            Path to generated PDF file
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, 
                Image, Table, TableStyle
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            # Generate filename
            pdf_filename = f"PPE_Violation_Report_{report_date.strftime('%Y-%m-%d')}.pdf"
            pdf_path = os.path.join(self.output_dir, pdf_filename)
            
            # Create document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=30
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#6b7280'),
                spaceAfter=20
            )
            
            # === Title ===
            story.append(Paragraph("PPE Compliance Daily Report", title_style))
            story.append(Paragraph(
                f"Report Date: {report_date.strftime('%B %d, %Y')}",
                subtitle_style
            ))
            story.append(Spacer(1, 0.2 * inch))
            
            # === Summary Table ===
            summary_data = [
                ["Metric", "Value"],
                ["Total Detections", str(stats['total_detections'])],
                ["Total Violations", str(stats['total_violations'])],
                ["Compliance Rate", f"{stats['compliance_rate']:.1f}%"],
                ["No Helmet Violations", str(stats.get('no_helmet_count', 0))],
                ["No Vest Violations", str(stats.get('no_vest_count', 0))],
                ["Both Missing", str(stats.get('both_missing_count', 0))],
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            story.append(Paragraph("Summary Statistics", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))
            story.append(summary_table)
            story.append(Spacer(1, 0.3 * inch))
            
            # === Hourly Chart ===
            chart_path = self._generate_hourly_chart(violations)
            if chart_path and os.path.exists(chart_path):
                story.append(Paragraph("Violation Timeline", styles['Heading2']))
                story.append(Spacer(1, 0.1 * inch))
                story.append(Image(chart_path, width=5*inch, height=3*inch))
                story.append(Spacer(1, 0.3 * inch))
            
            # === Violation Details ===
            story.append(Paragraph("Violation Details", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))
            
            violation_list = [v for v in violations if not (v.has_helmet and v.has_vest)]
            for i, violation in enumerate(violation_list[:10], 1):
                time_str = violation.timestamp.strftime('%H:%M:%S')
                type_str = violation.violation_type.replace('_', ' ').title()
                
                story.append(Paragraph(
                    f"<b>{i}.</b> {time_str} - {type_str} - Camera: {violation.camera_id}",
                    styles['Normal']
                ))
                
                # Include annotated image if exists
                if violation.annotated_image_path and os.path.exists(violation.annotated_image_path):
                    try:
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(Image(
                            violation.annotated_image_path,
                            width=4*inch,
                            height=3*inch
                        ))
                    except:
                        pass
                
                story.append(Spacer(1, 0.2 * inch))
            
            # === Footer ===
            story.append(Spacer(1, 0.5 * inch))
            story.append(Paragraph(
                "<i>This report was automatically generated by the PPE Detection System.</i>",
                styles['Normal']
            ))
            
            # Build PDF
            doc.build(story)
            print(f"✅ PDF generated: {pdf_path}")
            
            return pdf_path
            
        except ImportError as e:
            print(f"⚠️ ReportLab not available: {e}")
            return self._generate_text_report(report_date, violations, stats)
    
    def _generate_hourly_chart(self, violations: List) -> str:
        """Generate hourly violation distribution chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-GUI backend
            import matplotlib.pyplot as plt
            
            # Extract hours
            hours = [
                v.timestamp.hour 
                for v in violations 
                if not (v.has_helmet and v.has_vest)
            ]
            
            if not hours:
                return None
            
            plt.figure(figsize=(10, 6))
            plt.hist(hours, bins=24, range=(0, 24), edgecolor='black', color='#ef4444')
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Violation Count', fontsize=12)
            plt.title('Violations by Hour', fontsize=14)
            plt.xticks(range(0, 24, 2))
            plt.grid(axis='y', alpha=0.3)
            
            chart_path = os.path.join(self.output_dir, 'hourly_chart.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
            return None
    
    def _generate_text_report(
        self,
        report_date: date,
        violations: List,
        stats: Dict[str, Any]
    ) -> str:
        """Fallback text report when PDF libraries unavailable."""
        filename = f"PPE_Report_{report_date.strftime('%Y-%m-%d')}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("      PPE COMPLIANCE DAILY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {report_date.strftime('%B %d, %Y')}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Detections:  {stats['total_detections']}\n")
            f.write(f"Total Violations:  {stats['total_violations']}\n")
            f.write(f"Compliance Rate:   {stats['compliance_rate']:.1f}%\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("VIOLATION BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            f.write(f"No Helmet:    {stats.get('no_helmet_count', 0)}\n")
            f.write(f"No Vest:      {stats.get('no_vest_count', 0)}\n")
            f.write(f"Both Missing: {stats.get('both_missing_count', 0)}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("Auto-generated by PPE Detection System\n")
        
        return filepath
