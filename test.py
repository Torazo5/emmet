import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import getpass  # To safely prompt for the password
import os

print(os.getenv("EMAIL_PASS"))
def send_email(subject, body, to_email):
    # Gmail SMTP server configuration
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "tokud70460@gapps.uwcsea.edu.sg"  # Replace with your Gmail address

    # Prompt the user to enter the email password
    sender_password = os.getenv("EMAIL_PASS")

    try:
        # Setup the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Start TLS for security
        server.login(sender_email, sender_password)  # Login with the sender's email credentials

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the email body
        msg.attach(MIMEText(body, 'plain', 'utf-8'))  # UTF-8 encoding

        # Send the email
        server.send_message(msg)

        # Terminate the SMTP session
        server.quit()

        print(f"Email successfully sent to {to_email}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
send_email(
    subject="Test Email from Python",
    body="This is a test email sent using Python.",
    to_email="tokud70460@gapps.uwcsea.edu.sg"  # Replace with the recipient's email
)
