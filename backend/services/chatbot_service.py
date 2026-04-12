"""
Chatbot Service — Violation Query Assistant

Uses OpenAI API to convert natural language questions about PPE violations
into SQL queries against the local database, then returns conversational answers.

Part of the Intelligent PPE Compliance Monitoring System.
"""

import os
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import text
from sqlalchemy.orm import Session

# Try to import openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Database schema description for the LLM
DB_SCHEMA = """
You have access to an SQLite database with these tables:

TABLE: violations
- id (INTEGER, primary key)
- timestamp (DATETIME) — when violation was detected
- site_location (VARCHAR) — e.g. 'Construction Site A'
- camera_id (VARCHAR) — e.g. 'CAM-001'
- has_helmet (BOOLEAN) — True if helmet present
- has_vest (BOOLEAN) — True if vest present
- violation_type (VARCHAR) — 'no_helmet', 'no_vest', or 'both_missing'
- decision_path (VARCHAR) — 'Fast Safe', 'Fast Violation', 'Rescue Head', 'Rescue Body', 'Critical'
- detection_confidence (FLOAT) — 0.0 to 1.0
- sam_activated (BOOLEAN) — True if SAM verification was used
- processing_time_ms (FLOAT) — detection time in milliseconds
- report_sent (BOOLEAN) — True if included in daily report
- occurrence_count (INTEGER) — how many times re-detected in session
- total_duration_minutes (FLOAT) — total violation duration

TABLE: verified_violations
- id (INTEGER, primary key)
- timestamp (DATETIME) — when verified
- person_id (INTEGER) — tracked person ID
- camera_zone (VARCHAR) — e.g. 'zone_1'
- violation_type (VARCHAR) — 'no_helmet', 'no_vest', 'both_missing'
- judge_confirmed (BOOLEAN) — True if SAM Judge confirmed
- judge_confidence (FLOAT) — SAM confidence score
- sentry_confidence (FLOAT) — YOLO confidence score
- decision_path (VARCHAR)

TABLE: daily_reports
- id (INTEGER, primary key)
- report_date (DATE)
- total_detections (INTEGER)
- total_violations (INTEGER)
- compliance_rate (FLOAT) — percentage
- email_sent (BOOLEAN)
"""

SYSTEM_PROMPT = f"""You are a PPE Safety Compliance Assistant for a construction site monitoring system.

You help site managers understand violation data by converting their natural language questions into SQL queries.

{DB_SCHEMA}

RULES:
1. Generate ONLY SELECT queries. Never INSERT, UPDATE, DELETE, or DROP.
2. Always use SQLite-compatible syntax.
3. For date filtering, use date() and datetime() functions.
4. Keep queries simple and efficient.
5. When the user asks about "today", use date('now').
6. When the user asks about "this week", use date('now', '-7 days').
7. When the user asks about "this month", use date('now', 'start of month').
8. Return results in a friendly, conversational tone suitable for a site manager.
9. If the question cannot be answered from the database, say so politely.
10. Use the violations table primarily. Use verified_violations only when asked about SAM/Judge verified data.

Respond with JSON in this exact format:
{{
  "sql": "SELECT ... FROM ...",
  "explanation": "Brief explanation of what the query does",
  "answer_template": "Template for presenting results, use {{results}} placeholder"
}}

If no SQL is needed (e.g., greeting or off-topic), respond with:
{{
  "sql": null,
  "explanation": null,
  "answer_template": "Your conversational response here"
}}
"""


class ChatbotService:
    """
    Handles natural language queries about PPE violations
    using OpenAI for text-to-SQL conversion.
    """

    def __init__(self):
        self.client = None
        self.model = "gpt-4o-mini"  # Cost-effective for text-to-SQL

        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)

    def is_available(self) -> bool:
        """Check if the chatbot service is properly configured."""
        return self.client is not None

    async def ask(self, question: str, db: Session) -> Dict[str, Any]:
        """
        Process a natural language question about violations.

        Args:
            question: User's natural language question
            db: SQLAlchemy database session

        Returns:
            Dict with answer, data, and metadata
        """
        if not self.is_available():
            return {
                "answer": "⚠️ Chatbot is not configured. Please set OPENAI_API_KEY in your .env file.",
                "data": None,
                "sql": None,
                "success": False
            }

        try:
            # Step 1: Ask OpenAI to generate SQL
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,  # Low temperature for consistent SQL
                response_format={"type": "json_object"}
            )

            raw_response = response.choices[0].message.content
            parsed = json.loads(raw_response)

            sql_query = parsed.get("sql")
            explanation = parsed.get("explanation")
            answer_template = parsed.get("answer_template", "")

            # Step 2: If no SQL needed (greeting, etc.)
            if not sql_query:
                return {
                    "answer": answer_template,
                    "data": None,
                    "sql": None,
                    "success": True
                }

            # Step 3: Safety check — only allow SELECT
            sql_upper = sql_query.strip().upper()
            if not sql_upper.startswith("SELECT"):
                return {
                    "answer": "🚫 I can only run read-only queries for safety.",
                    "data": None,
                    "sql": sql_query,
                    "success": False
                }

            # Step 4: Execute the query
            result = db.execute(text(sql_query))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]

            # Step 5: Generate a human-friendly answer
            answer = self._format_answer(question, rows, columns, explanation, answer_template)

            return {
                "answer": answer,
                "data": rows[:50],  # Limit to 50 rows
                "sql": sql_query,
                "total_rows": len(rows),
                "success": True
            }

        except json.JSONDecodeError:
            return {
                "answer": "❌ I had trouble understanding the response. Please try rephrasing your question.",
                "data": None,
                "sql": None,
                "success": False
            }
        except Exception as e:
            return {
                "answer": f"❌ Error processing your question: {str(e)}",
                "data": None,
                "sql": None,
                "success": False
            }

    def _format_answer(
        self,
        question: str,
        rows: List[Dict],
        columns: List[str],
        explanation: str,
        template: str
    ) -> str:
        """Format query results into a human-readable answer."""
        if not rows:
            return "📊 No data found matching your query. The database might be empty or the time range has no violations."

        # Simple formatting for common patterns
        if len(rows) == 1 and len(columns) == 1:
            # Single value result (e.g., COUNT, AVG)
            value = list(rows[0].values())[0]
            col_name = columns[0].lower()

            if "count" in col_name:
                return f"📊 **{value}** violations found based on your query."
            elif "rate" in col_name or "compliance" in col_name:
                return f"📊 The compliance rate is **{value:.1f}%**."
            elif "avg" in col_name:
                return f"📊 The average is **{value:.2f}**."
            else:
                return f"📊 Result: **{value}**"

        # Multiple rows — summarize
        if len(rows) <= 5:
            lines = [f"📊 Found **{len(rows)}** result(s):\n"]
            for i, row in enumerate(rows, 1):
                parts = [f"{k}: {v}" for k, v in row.items()]
                lines.append(f"  {i}. {' | '.join(parts)}")
            return "\n".join(lines)
        else:
            return f"📊 Found **{len(rows)}** results. Showing first 5:\n" + "\n".join(
                f"  {i+1}. {' | '.join(f'{k}: {v}' for k, v in row.items())}"
                for i, row in enumerate(rows[:5])
            )


# Global instance
_chatbot_service: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    """Get or create global chatbot service instance."""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service
