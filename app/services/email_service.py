"""
OrthoLink Email Service
Send emails via Resend API.
HC-7: No AI provider names in customer-facing communications.
"""

import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    """Send transactional emails via Resend."""

    def __init__(self) -> None:
        self._client = None

    def is_configured(self) -> bool:
        """Return True if Resend API key is set."""
        return bool(get_settings().resend_api_key)

    def _get_client(self):
        """Lazy-initialize Resend client."""
        if self._client is None:
            settings = get_settings()
            if settings.resend_api_key:
                import resend
                resend.api_key = settings.resend_api_key
                self._client = resend
            else:
                logger.warning("Resend API key not configured. Emails will be logged only.")
        return self._client

    def send_welcome_email(self, to_email: str, org_name: str, magic_link: str) -> bool:
        """Send welcome email to new organization admin."""
        subject = f"Welcome to OrthoLink - {org_name}"
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #1a202c;">Welcome to OrthoLink</h1>
            <p>Your organization <strong>{org_name}</strong> has been provisioned.</p>
            <p>Click the link below to access your regulatory intelligence dashboard:</p>
            <a href="{magic_link}" style="
                display: inline-block;
                background: #0ea5e9;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 6px;
                margin: 16px 0;
            ">Access Dashboard</a>
            <p style="color: #718096; font-size: 14px;">
                This link will expire in 24 hours. If you need a new link, contact your administrator.
            </p>
            <hr style="border: 1px solid #e2e8f0; margin: 24px 0;">
            <p style="color: #a0aec0; font-size: 12px;">
                OrthoLink - Regulatory Intelligence for Medical Devices
            </p>
        </div>
        """
        return self._send(to_email, subject, html)

    def send_alert_notification(
        self,
        to_email: str,
        country: str,
        regulation_name: str,
        severity: str,
        summary: str,
    ) -> bool:
        """Send regulatory change alert email."""
        subject = f"[OrthoLink Alert] {severity}: {regulation_name} ({country})"
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #1a202c;">Regulatory Change Alert</h2>
            <div style="background: #f7fafc; padding: 16px; border-radius: 8px; border-left: 4px solid
                {'#ef4444' if severity == 'Critical' else '#f59e0b' if severity == 'Major' else '#0ea5e9'};">
                <p><strong>Country:</strong> {country}</p>
                <p><strong>Regulation:</strong> {regulation_name}</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Summary:</strong> {summary}</p>
            </div>
            <p style="margin-top: 16px;">Log in to your dashboard for full details and impact analysis.</p>
        </div>
        """
        return self._send(to_email, subject, html)

    def _send(self, to_email: str, subject: str, html: str) -> bool:
        """Send an email. Returns True on success."""
        settings = get_settings()
        client = self._get_client()

        if client is None:
            logger.info(f"EMAIL (not sent - no API key): to={to_email} subject={subject}")
            return False

        try:
            client.Emails.send({
                "from": settings.resend_from_email,
                "to": to_email,
                "subject": subject,
                "html": html,
            })
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False


_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
