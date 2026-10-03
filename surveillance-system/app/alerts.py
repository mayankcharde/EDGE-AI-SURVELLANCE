import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import requests

from config import (
    EMAIL_ENABLED, SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO,
    TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)
from app.database import log_alert

logger = logging.getLogger(__name__)


def send_email(subject, body, snapshot_path=None):
    if not EMAIL_ENABLED:
        return False
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL_TO
        msg.attach(MIMEText(body, "plain"))

        if snapshot_path:
            try:
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-Disposition", "attachment",
                                   filename="snapshot.jpg")
                    msg.attach(img)
            except Exception as e:
                logger.warning(f"Could not attach snapshot: {e}")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info(f"Email sent to {ALERT_EMAIL_TO}")
        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False


def send_telegram(message, snapshot_path=None):
    if not TELEGRAM_ENABLED:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)

        if snapshot_path:
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(snapshot_path, "rb") as f:
                requests.post(url_photo,
                              data={"chat_id": TELEGRAM_CHAT_ID},
                              files={"photo": f},
                              timeout=15)
        return True
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


def dispatch_alert(event_id, person, event_type, explanation, snapshot_path=None):
    if person != "Unknown" and event_type == "KNOWN_PERSON":
        logger.info(f"Skipping alert dispatch for routine known person detection: {person}")
        return

    subject = f"[Surveillance] {event_type} - {person}"
    body = (
        f"Event: {event_type}\n"
        f"Person: {person}\n"
        f"Time: {__import__('datetime').datetime.now().isoformat()}\n\n"
        f"Explanation: {explanation}\n"
    )

    if EMAIL_ENABLED:
        ok = send_email(subject, body, snapshot_path)
        log_alert(event_id, "email", "sent" if ok else "failed")

    if TELEGRAM_ENABLED:
        ok = send_telegram(f"{subject}\n\n{explanation}", snapshot_path)
        log_alert(event_id, "telegram", "sent" if ok else "failed")