"""
PDF export functionality for RedVox reports.
Generates professional PDF reports with charts, tables, and visualizations.
"""

import io
import base64
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class PDFReportGenerator:
    """Generate PDF reports from RedVox data."""
    
    def __init__(self):
        self.reportlab_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if reportlab is available."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            self.reportlab_available = True
            self.pagesize = letter
            self.styles = getSampleStyleSheet()
            self.inch = inch
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.Table = Table
            self.TableStyle = TableStyle
            self.Image = Image
            self.colors = colors
            self.TA_CENTER = TA_CENTER
            self.TA_LEFT = TA_LEFT
        except ImportError:
            self.reportlab_available = False
    
    def generate_pdf(self, report_data: Dict, output_path: str) -> bool:
        """
        Generate PDF report from report data.
        
        Args:
            report_data: Dictionary with report content
            output_path: Output PDF file path
        
        Returns:
            True if successful, False otherwise
        """
        if not self.reportlab_available:
            print("reportlab not installed. Install with: pip install reportlab")
            return False
        
        try:
            doc = self.SimpleDocTemplate(
                output_path,
                pagesize=self.pagesize,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            title_style = self.styles['Heading1']
            title_style.alignment = self.TA_CENTER
            title = self.Paragraph("RedVox Scientific Data Report", title_style)
            story.append(title)
            story.append(self.Spacer(1, 12))
            
            # Report metadata
            metadata_style = self.styles['Normal']
            report_id = report_data.get('report_id', 'N/A')
            timestamp = report_data.get('timestamp', datetime.now().isoformat())
            
            metadata_text = f"""
            <b>Report ID:</b> {report_id}<br/>
            <b>Generated:</b> {timestamp}<br/>
            <b>Station ID:</b> {report_data.get('station_id', 'N/A')}<br/>
            <b>Device:</b> {report_data.get('device', 'N/A')}
            """
            metadata = self.Paragraph(metadata_text, metadata_style)
            story.append(metadata)
            story.append(self.Spacer(1, 24))
            
            # Executive Summary
            if 'executive_summary' in report_data:
                heading = self.Paragraph("Executive Summary", self.styles['Heading2'])
                story.append(heading)
                story.append(self.Spacer(1, 12))
                
                summary = self.Paragraph(report_data['executive_summary'], self.styles['Normal'])
                story.append(summary)
                story.append(self.Spacer(1, 12))
            
            # Key Metrics
            if 'key_metrics' in report_data:
                heading = self.Paragraph("Key Metrics", self.styles['Heading2'])
                story.append(heading)
                story.append(self.Spacer(1, 12))
                
                metrics = report_data['key_metrics']
                metric_data = [['Metric', 'Value']]
                for key, value in metrics.items():
                    metric_data.append([key, str(value)])
                
                table = self.Table(metric_data, colWidths=[3*inch, 2*inch])
                table.setStyle(self.TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), self.TA_LEFT),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, self.colors.black)
                ]))
                story.append(table)
                story.append(self.Spacer(1, 24))
            
            # Sensor Analysis
            if 'sensor_analysis' in report_data:
                heading = self.Paragraph("Sensor Analysis", self.styles['Heading2'])
                story.append(heading)
                story.append(self.Spacer(1, 12))
                
                sensor_analysis = report_data['sensor_analysis']
                for sensor_name, sensor_data in sensor_analysis.items():
                    sensor_heading = self.Paragraph(f"<b>{sensor_name}</b>", self.styles['Heading3'])
                    story.append(sensor_heading)
                    story.append(self.Spacer(1, 6))
                    
                    sensor_text = self.Paragraph(str(sensor_data), self.styles['Normal'])
                    story.append(sensor_text)
                    story.append(self.Spacer(1, 12))
            
            # Visualizations
            if 'visualizations' in report_data:
                heading = self.Paragraph("Visualizations", self.styles['Heading2'])
                story.append(heading)
                story.append(self.Spacer(1, 12))
                
                for viz_name, viz_data in report_data['visualizations'].items():
                    if isinstance(viz_data, str) and viz_data.startswith('data:image'):
                        # Base64 encoded image
                        # Remove data URL prefix
                        b64_data = viz_data.split(',')[1]
                        image_data = base64.b64decode(b64_data)
                        
                        # Create image from bytes
                        img_io = io.BytesIO(image_data)
                        img = self.Image(img_io, width=5*inch, height=3*inch)
                        story.append(img)
                        story.append(self.Spacer(1, 12))
            
            # Recommendations
            if 'recommendations' in report_data:
                heading = self.Paragraph("Recommendations", self.styles['Heading2'])
                story.append(heading)
                story.append(self.Spacer(1, 12))
                
                recommendations = report_data['recommendations']
                for i, rec in enumerate(recommendations, 1):
                    rec_text = self.Paragraph(f"{i}. {rec}", self.styles['Normal'])
                    story.append(rec_text)
                    story.append(self.Spacer(1, 6))
            
            # Insights
            if 'insights' in report_data:
                heading = self.Paragraph("Key Insights", self.styles['Heading2'])
                story.append(heading)
                story.append(self.Spacer(1, 12))
                
                insights = report_data['insights']
                for insight in insights:
                    insight_text = self.Paragraph(f"• {insight}", self.styles['Normal'])
                    story.append(insight_text)
                    story.append(self.Spacer(1, 6))
            
            # Footer
            story.append(self.Spacer(1, 24))
            footer = self.Paragraph(
                "Generated by RedVox Scientific Data Platform",
                self.styles['Normal']
            )
            footer.alignment = self.TA_CENTER
            story.append(footer)
            
            # Build PDF
            doc.build(story)
            
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False
    
    def generate_pdf_from_html(self, html_content: str, output_path: str) -> bool:
        """
        Generate PDF from HTML content (alternative method).
        
        Args:
            html_content: HTML string
            output_path: Output PDF file path
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import weasyprint
            weasyprint.HTML(string=html_content).write_pdf(output_path)
            return True
        except ImportError:
            print("weasyprint not installed. Install with: pip install weasyprint")
            return False
        except Exception as e:
            print(f"Error generating PDF from HTML: {e}")
            return False


def export_report_to_pdf(report_generator, report, output_path: str) -> bool:
    """
    Export report to PDF format.
    
    Args:
        report_generator: ReportGenerator instance
        report: DashboardReport object
        output_path: Output PDF file path
    
    Returns:
        True if successful, False otherwise
    """
    pdf_gen = PDFReportGenerator()
    
    # Convert report to dictionary mapping from DashboardReport structure
    report_data = {
        'report_id': getattr(report, 'report_id', 'N/A'),
        'timestamp': getattr(report, 'generated_at', datetime.now().isoformat()),
        'station_id': report.data_period.get('station_id', 'N/A') if hasattr(report, 'data_period') else 'N/A',
        'device': 'RedVox',
        'executive_summary': getattr(report, 'executive_summary', ''),
        'key_metrics': getattr(report, 'key_metrics', {}),
        'sensor_analysis': {s.sensor_type: s.key_findings for s in report.sensor_insights} if hasattr(report, 'sensor_insights') else {},
        'visualizations': getattr(report, 'visualizations', {}),
        'recommendations': getattr(report, 'actionable_recommendations', []),
        'insights': []
    }
    
    return pdf_gen.generate_pdf(report_data, output_path)
