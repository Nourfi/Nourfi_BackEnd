import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = (os.getenv("GMAIL_ADDRESS") or "").strip()
GMAIL_APP_PASSWORD = (os.getenv("GMAIL_APP_PASSWORD") or "").strip().replace(" ", "")
def send_email(to_email: str, subject: str, html_content: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())


def send_reset_password_email(to_email: str, reset_token: str):
    reset_link = f"http://localhost:5173/reset-password?token={reset_token}"
    html = f"""
        <p>Bonjour,</p>
        <p>Vous avez demandé à réinitialiser votre mot de passe.</p>
        <p><a href="{reset_link}">Cliquez ici pour créer un nouveau mot de passe</a></p>
        <p>Ce lien expire dans 30 minutes. Si vous n'avez pas fait cette demande, ignorez cet e-mail.</p>
    """
    send_email(to_email, "Réinitialisation de votre mot de passe", html)


def send_verification_email(to_email: str, code: str):
    html = f"""
        <p>Bonjour,</p>
        <p>Voici votre code de vérification NourFi :</p>
        <h2 style="letter-spacing: 4px;">{code}</h2>
        <p>Ce code expire dans 15 minutes.</p>
    """
    send_email(to_email, "Vérifiez votre adresse e-mail - NourFi", html)