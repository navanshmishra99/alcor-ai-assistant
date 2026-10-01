from __future__ import annotations

import logging
import os
import re
import smtplib
from email.message import EmailMessage

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool


logger = logging.getLogger("ask_alcor.leads")

router = APIRouter()


class LeadRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    message: str = Field(default="", max_length=2000)
    question: str = Field(default="", max_length=1000)
    website: str = Field(default="", max_length=500)

    @field_validator("name", "email", "message", "question")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Enter a valid email address.")
        return value


def _send_lead_email(lead: LeadRequest) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip()
    recipient = os.getenv("LEAD_NOTIFY_TO", "").strip()

    if not host or host == "your-smtp-server" or not sender or not recipient:
        raise HTTPException(
            status_code=503,
            detail="Lead submission is temporarily unavailable.",
        )

    email = EmailMessage()
    email["Subject"] = "Ask Alcor contact request"
    email["From"] = sender
    email["To"] = recipient
    email["Reply-To"] = lead.email
    email.set_content(
        f"Name: {lead.name}\n"
        f"Email: {lead.email}\n"
        f"Question: {lead.question or '(not provided)'}\n\n"
        f"Message:\n{lead.message or '(not provided)'}\n"
    )

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")

    try:
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls()
            if username:
                smtp.login(username, password)
            smtp.send_message(email)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        logger.warning("Lead notification could not be sent: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Lead submission is temporarily unavailable.",
        ) from exc


@router.post("/leads", status_code=202)
async def submit_lead(lead: LeadRequest) -> dict[str, str]:
    if lead.website:
        return {"status": "received"}

    await run_in_threadpool(_send_lead_email, lead)
    return {"status": "received"}