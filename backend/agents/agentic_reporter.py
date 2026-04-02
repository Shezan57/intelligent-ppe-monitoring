"""
Agentic Reporter — Module 4

End-of-day (or end-of-session) report generator that:
1. Queries `verified_violations` DB table for Judge-confirmed violations
2. Groups violations by person_id to identify serial offenders
3. Sends structured data to an LLM API for professional OSHA-style summary
4. Generates a PDF with:
   - LLM-generated summary
   - Violation table (by person)
   - Embedded ROI evidence images

Can run manually or be scheduled via APScheduler.
"""

import os
import json
import time
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from collections import defaultdict

from config.settings import settings

logger = logging.getLogger(__name__)

# ── LLM prompt for OSHA safety officer report ────────────────────────────────
OSHA_PROMPT = """Act as an OSHA Safety Officer. Review this list of PPE violations
detected at a construction site. Write a professional daily safety summary.

For each person, identify whether they are a serial offender (multiple violations).
Reference standard construction safety protocols (OSHA 1926 Subpart E).

Be concise but authoritative. Use bullet points for clarity.

VIOLATIONS DATA:
{violations_json}

STATISTICS:
- Total verified violations: {total_violations}
- Unique workers in violation: {unique_workers}
- Most common violation type: {most_common}
- Date: {report_date}

Write the summary now:"""


class AgenticReporter:
    """
    LLM-powered report generator for verified PPE violations.

    Queries Judge-confirmed violations from the database,
    generates an LLM summary, and produces a PDF report.
    """

    def __init__(
        self,
        output_dir: str = None,
        llm_provider: str = "google",  # "google", "openai", or "none"
        api_key: str = None,
    ):
        """
        Initialize the reporter.

        Args:
            output_dir: Directory for output PDFs
            llm_provider: LLM to use for summary generation
            api_key: API key for the LLM provider
        """
        self.output_dir = output_dir or getattr(settings, 'report_output_dir', 'reports')
        self.llm_provider = llm_provider
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(
        self,
        target_date: Optional[date] = None,
        db_session_factory=None,
    ) -> Dict[str, Any]:
        """
        Generate a complete end-of-day report.

        Args:
            target_date: Date to report on (default: today)
            db_session_factory: SQLAlchemy sessionmaker

        Returns:
            Report result dict with pdf_path, stats, etc.
        """
        if target_date is None:
            target_date = date.today()

        print(f"\n{'='*50}")
        print(f"AGENTIC REPORTER — {target_date}")
        print(f"{'='*50}")

        # Step 1: Query verified violations
        violations = self._query_violations(target_date, db_session_factory)
        if not violations:
            print("  No verified violations found for this date.")
            return {"success": True, "message": "No violations", "total": 0}

        # Step 2: Aggregate by person
        person_groups = self._group_by_person(violations)
        stats = self._calculate_stats(violations, person_groups)

        print(f"  Found {len(violations)} verified violations for {len(person_groups)} workers")

        # Step 3: Generate LLM summary
        llm_summary = self._generate_llm_summary(violations, person_groups, stats, target_date)

        # Step 4: Generate PDF
        pdf_path = self._generate_pdf(target_date, violations, person_groups, stats, llm_summary)

        print(f"  PDF saved: {pdf_path}")
        print(f"{'='*50}\n")

        return {
            "success": True,
            "report_date": str(target_date),
            "pdf_path": pdf_path,
            "total_violations": len(violations),
            "unique_workers": len(person_groups),
            "stats": stats,
            "llm_summary": llm_summary[:200] + "..." if len(llm_summary) > 200 else llm_summary,
        }

    def _query_violations(self, target_date: date, db_session_factory) -> list:
        """Query verified_violations table for the given date."""
        if db_session_factory is None:
            logger.warning("No DB session factory provided")
            return []

        try:
            from database.models import VerifiedViolation
            session = db_session_factory()

            violations = session.query(VerifiedViolation).filter(
                VerifiedViolation.timestamp >= datetime.combine(target_date, datetime.min.time()),
                VerifiedViolation.timestamp <= datetime.combine(target_date, datetime.max.time()),
            ).order_by(VerifiedViolation.person_id).all()

            # Convert to dicts for serialization
            result = []
            for v in violations:
                result.append({
                    "id": v.id,
                    "timestamp": v.timestamp.isoformat() if v.timestamp else "",
                    "person_id": v.person_id,
                    "violation_type": v.violation_type,
                    "image_path": v.image_path,
                    "camera_zone": v.camera_zone,
                    "judge_confidence": v.judge_confidence,
                    "decision_path": v.decision_path,
                    "sentry_confidence": v.sentry_confidence,
                    "person_bbox": v.person_bbox,
                })

            session.close()
            return result

        except Exception as e:
            logger.error(f"DB query error: {e}")
            return []

    def _group_by_person(self, violations: list) -> Dict[int, list]:
        """Group violations by person_id to find serial offenders."""
        groups = defaultdict(list)
        for v in violations:
            groups[v["person_id"]].append(v)
        return dict(groups)

    def _calculate_stats(self, violations: list, groups: dict) -> dict:
        """Calculate report statistics."""
        type_counts = defaultdict(int)
        for v in violations:
            type_counts[v["violation_type"]] += 1

        most_common = max(type_counts, key=type_counts.get) if type_counts else "none"

        serial_offenders = [pid for pid, vlist in groups.items() if len(vlist) > 1]

        return {
            "total_violations": len(violations),
            "unique_workers": len(groups),
            "type_counts": dict(type_counts),
            "most_common_type": most_common,
            "serial_offenders": serial_offenders,
            "serial_offender_count": len(serial_offenders),
        }

    def _generate_llm_summary(
        self, violations: list, groups: dict, stats: dict, target_date: date
    ) -> str:
        """
        Generate professional summary using LLM API.
        Falls back to template-based summary if no API key.
        """
        # Prepare violations data for LLM
        violations_for_llm = []
        for pid, vlist in groups.items():
            violations_for_llm.append({
                "person_id": pid,
                "violation_count": len(vlist),
                "types": list(set(v["violation_type"] for v in vlist)),
                "camera_zones": list(set(v["camera_zone"] for v in vlist)),
            })

        prompt = OSHA_PROMPT.format(
            violations_json=json.dumps(violations_for_llm, indent=2),
            total_violations=stats["total_violations"],
            unique_workers=stats["unique_workers"],
            most_common=stats["most_common_type"],
            report_date=str(target_date),
        )

        # Try LLM API
        if self.api_key:
            try:
                summary = self._call_llm(prompt)
                if summary:
                    print("  LLM summary generated successfully")
                    return summary
            except Exception as e:
                logger.warning(f"LLM API error: {e}")
                print(f"  LLM API error: {e}, using template fallback")

        # Fallback: template-based summary
        return self._template_summary(groups, stats, target_date)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API for summary generation."""
        if self.llm_provider == "google":
            return self._call_gemini(prompt)
        elif self.llm_provider == "openai":
            return self._call_openai(prompt)
        return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            logger.warning("google-generativeai not installed")
            return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return response.choices[0].message.content
        except ImportError:
            logger.warning("openai not installed")
            return None

    def _template_summary(self, groups: dict, stats: dict, target_date: date) -> str:
        """Template-based fallback summary (no LLM required)."""
        lines = [
            f"DAILY PPE COMPLIANCE REPORT - {target_date}",
            f"{'='*50}",
            "",
            f"Total verified violations: {stats['total_violations']}",
            f"Workers in violation: {stats['unique_workers']}",
            f"Most common violation: {stats['most_common_type']}",
            "",
            "WORKER VIOLATION SUMMARY:",
        ]

        for pid, vlist in sorted(groups.items()):
            types = set(v["violation_type"] for v in vlist)
            serial = " [SERIAL OFFENDER]" if len(vlist) > 1 else ""
            lines.append(f"  Worker ID {pid}: {len(vlist)} violation(s) - {', '.join(types)}{serial}")

        if stats["serial_offenders"]:
            lines.append("")
            lines.append(f"SERIAL OFFENDERS: Workers {stats['serial_offenders']}")
            lines.append("  Recommendation: Immediate supervisor intervention required.")

        lines.append("")
        lines.append("Reference: OSHA 1926 Subpart E - Personal Protective and Life Saving Equipment")
        lines.append("Action Required: All violations must be corrected before workers return to site.")

        return "\n".join(lines)

    def _generate_pdf(
        self,
        target_date: date,
        violations: list,
        groups: dict,
        stats: dict,
        llm_summary: str,
    ) -> str:
        """Generate PDF report with summary, table, and evidence images."""
        filename = f"PPE_Report_{target_date.strftime('%Y-%m-%d')}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch, cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

            doc = SimpleDocTemplate(filepath, pagesize=A4,
                                     topMargin=1*cm, bottomMargin=1*cm,
                                     leftMargin=1.5*cm, rightMargin=1.5*cm)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, spaceAfter=12)
            story.append(Paragraph(f"PPE Compliance Report - {target_date}", title_style))
            story.append(Spacer(1, 12))

            # Stats summary
            stats_text = (
                f"<b>Total Violations:</b> {stats['total_violations']} | "
                f"<b>Workers:</b> {stats['unique_workers']} | "
                f"<b>Serial Offenders:</b> {stats['serial_offender_count']}"
            )
            story.append(Paragraph(stats_text, styles['Normal']))
            story.append(Spacer(1, 12))

            # LLM Summary section
            story.append(Paragraph("<b>Safety Officer Summary</b>", styles['Heading2']))
            story.append(Spacer(1, 6))
            for line in llm_summary.split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['Normal']))
                    story.append(Spacer(1, 3))
            story.append(Spacer(1, 12))

            # Violation table
            story.append(Paragraph("<b>Violation Details</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            table_data = [["Person ID", "Type", "Time", "Zone", "Confidence"]]
            for v in violations:
                ts = v["timestamp"][:19] if v["timestamp"] else "N/A"
                conf = f"{v.get('judge_confidence', 0):.2f}" if v.get('judge_confidence') else "N/A"
                table_data.append([
                    str(v["person_id"]),
                    v["violation_type"],
                    ts,
                    v.get("camera_zone", "N/A"),
                    conf,
                ])

            table = Table(table_data, colWidths=[60, 80, 120, 60, 60])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))

            # Evidence images (first 10 to avoid huge PDFs)
            story.append(Paragraph("<b>Evidence (ROI Crops)</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            img_count = 0
            for v in violations[:10]:
                img_path = v.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    try:
                        img = RLImage(img_path, width=3*inch, height=2*inch)
                        caption = f"Person {v['person_id']} - {v['violation_type']} ({v['timestamp'][:19]})"
                        story.append(Paragraph(caption, styles['Normal']))
                        story.append(img)
                        story.append(Spacer(1, 8))
                        img_count += 1
                    except Exception:
                        pass

            if img_count == 0:
                story.append(Paragraph("No ROI evidence images available.", styles['Normal']))

            # Footer
            story.append(Spacer(1, 20))
            story.append(Paragraph(
                "Reference: OSHA 1926 Subpart E. Generated by Sentry-Judge PPE Monitoring System.",
                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey)
            ))

            doc.build(story)
            print(f"  PDF generated: {filepath}")

        except ImportError:
            # Fallback: text file
            filepath = filepath.replace('.pdf', '.txt')
            with open(filepath, 'w') as f:
                f.write(llm_summary)
                f.write("\n\nVIOLATIONS:\n")
                for v in violations:
                    f.write(f"  Person {v['person_id']}: {v['violation_type']} at {v['timestamp']}\n")
            print(f"  Text report generated (ReportLab not available): {filepath}")

        return filepath
