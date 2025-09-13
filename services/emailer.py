import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional, Union
import sqlite3
import certifi

def send_email(settings_row: Union[sqlite3.Row, dict], subject: str, body: str, html_body: Optional[str] = None) -> bool:
    """Send an email using SMTP settings from the given row or dict.

    Returns ``True`` if an email was sent, ``False`` if required settings are
    missing.  Raises ``Exception`` for SMTP failures.
    """
    user = (settings_row.get("smtp_user") or "").strip()  # type: ignore[arg-type]
    pwd = (settings_row.get("smtp_pass") or "").replace(" ", "").strip()  # type: ignore[arg-type]
    recips_raw = settings_row.get("recipients") or ""  # type: ignore[arg-type]
    if isinstance(recips_raw, str):
        recips = [r.strip() for r in recips_raw.split(",") if r.strip()]
    else:
        recips = list(recips_raw)
    if not user or not pwd or not recips:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(recips)
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
            server.login(user, pwd)
            server.send_message(msg)
    except ssl.SSLError:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(user, pwd)
            server.send_message(msg)
    return True
