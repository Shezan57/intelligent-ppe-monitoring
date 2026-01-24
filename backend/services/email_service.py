"""
Email Service

Sends automated emails with PDF attachments to managers.
Supports SMTP (Gmail, Outlook, etc.).
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import date
from typing import List, Dict, Any

from config.settings import settings


class EmailService:
    """
    Sends automated emails with PDF attachments.
    
    Configured via environment variables for SMTP credentials.
    Supports Gmail (with app passwords) and other SMTP servers.
    """
    
    def __init__(self):
        """Initialize email service with settings."""
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.sender_email = settings.sender_email
        self.sender_password = settings.sender_password
    
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.sender_email and self.sender_password)
    
    def send_daily_report(
        self,
        recipients: List[str],
        report_date: date,
        pdf_path: str,
        summary_stats: Dict[str, Any]
    ) -> bool:
        """
        Send daily report email with PDF attachment.
        
        Args:
            recipients: List of manager email addresses
            report_date: Date of the report
            pdf_path: Path to PDF file to attach
            summary_stats: Statistics for email body
            
        Returns:
            True if email sent successfully
        """
        if not self.is_configured():
            print("⚠️ Email not configured - skipping")
            return False
        
        if not recipients:
            print("⚠️ No recipients specified")
            return False
        
        if not os.path.exists(pdf_path):
            print(f"⚠️ PDF file not found: {pdf_path}")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"PPE Compliance Report - {report_date.strftime('%Y-%m-%d')}"
            
            # Email body
            body = self._create_email_body(report_date, summary_stats)
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, 'rb') as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                pdf_attachment.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=os.path.basename(pdf_path)
                )
                msg.attach(pdf_attachment)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✅ Report emailed to {len(recipients)} recipient(s)")
            return True
            
        except smtplib.SMTPAuthenticationError:
            print("❌ Email authentication failed - check credentials")
            return False
            
        except smtplib.SMTPException as e:
            print(f"❌ Email sending failed: {e}")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected email error: {e}")
            return False
    
    def _create_email_body(
        self,
        report_date: date,
        stats: Dict[str, Any]
    ) -> str:
        """Create email body text."""
        return f"""Dear Safety Manager,

Please find attached the daily PPE compliance report for {report_date.strftime('%B %d, %Y')}.

SUMMARY
{'='*40}
Total Detections:  {stats.get('total_detections', 0)}
Total Violations:  {stats.get('total_violations', 0)}
Compliance Rate:   {stats.get('compliance_rate', 100):.1f}%

VIOLATION BREAKDOWN
{'='*40}
No Helmet:         {stats.get('no_helmet_count', 0)}
No Vest:           {stats.get('no_vest_count', 0)}
Both Missing:      {stats.get('both_missing_count', 0)}

Detailed evidence and analysis are included in the attached PDF report.

---
This is an automated message from the PPE Detection System.
Please do not reply to this email.

Best regards,
Automated Safety Monitoring System
"""
    
    def send_test_email(self, recipient: str) -> bool:
        """
        Send a test email to verify configuration.
        
        Args:
            recipient: Email address to send test to
            
        Returns:
            True if test email sent successfully
        """
        if not self.is_configured():
            print("⚠️ Email not configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = "PPE Detection System - Test Email"
            
            body = """This is a test email from the PPE Detection System.

If you received this email, your email configuration is working correctly.

Best regards,
PPE Detection System
"""
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✅ Test email sent to {recipient}")
            return True
            
        except Exception as e:
            print(f"❌ Test email failed: {e}")
            return False
