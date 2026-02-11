import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv


def send_mail(content: str, receiver: str):
    """Sends an email"""

    load_dotenv()
    password = os.getenv("PASSWORD")

    msg = EmailMessage()
    msg["From"] = "ardian21ardian@gmail.com"
    msg["To"] = receiver
    msg["Subject"] = "Notification"
    msg.set_content(content)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login("ardian21ardian@gmail.com", password)
        server.send_message(msg)


def notify_mail(content: str, receiver="all"):
    """Sends an email notification

    If receiver is "all", the email will be sent to all the users in the database. Else, specify the email of the receiver.
    """

    if receiver == "all":
        send_mail(content, "adrianlanchares@alu.comillas.edu")
        send_mail(content, "sergio.herreros@alu.comillas.edu")
        send_mail(content, "ibayon@alu.comillas.edu")
    else:
        send_mail(content, receiver)
