import re

# Match standard email addresses
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)

# Match Australian and international phone number formats
PHONE_PATTERN = re.compile(
    r"(?<!\d)(\+?61[\s\-]?)?0?[2-9]\d[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)"
    r"|(?<!\d)(\+?1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}(?!\d)"
    r"|(?<!\d)\+\d{1,3}[\s\-]?\d{6,14}(?!\d)"
)


def anonymize_resume_text(text: str, employee_name: str = "") -> str:
    """
    Remove PII from resume text before sending to any AI API.
    Replaces name, email, and phone with placeholder tokens.
    The original cache text is never modified — only the AI copy is anonymized.
    """
    if not text:
        return text

    # Replace full name and each individual name part (skip parts <= 2 chars
    # to avoid accidentally replacing common short words)
    if employee_name and employee_name.strip():
        parts = employee_name.strip().split()

        escaped_full = re.escape(employee_name.strip())
        text = re.sub(
            r"(?<![a-zA-Z])" + escaped_full + r"(?![a-zA-Z])",
            "[CANDIDATE]",
            text,
            flags=re.IGNORECASE
        )

        for part in parts:
            if len(part) > 2:
                escaped_part = re.escape(part)
                text = re.sub(
                    r"(?<![a-zA-Z])" + escaped_part + r"(?![a-zA-Z])",
                    "[CANDIDATE]",
                    text,
                    flags=re.IGNORECASE
                )

    # Replace all email addresses
    text = EMAIL_PATTERN.sub("[EMAIL]", text)

    # Replace all phone numbers
    text = PHONE_PATTERN.sub("[PHONE]", text)

    return text