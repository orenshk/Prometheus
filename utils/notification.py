import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pushbullet import Pushbullet
from utils.config import config_from_file

def push_note(subject, body):
    config = config_from_file(key='pushbullet')
    pb = Pushbullet(api_key=config['api_key'])
    for channel in pb.channels:
        if channel.channel_tag == config['channel']:
            channel.push_note(subject, body)

def email(to_addrs, subject, body, service='gmail'):
    config = config_from_file(key='gmail')
    if service == 'gmail':
        _gmail(gmail_addr=config['username'], to_addrs=to_addrs, subject=subject, body=body, password=config['password'])
    else:
        print('Unsupported mail service {}'.format(service))


def _gmail(gmail_addr, to_addrs, subject, body, password):
    msg = MIMEMultipart()

    msg['From'] =  gmail_addr
    if isinstance(to_addrs, str):
        msg['To'] = to_addrs
    else:
        msg['To'] = ','.join(to_addrs)

    msg['Subject'] = subject

    # add body
    msg.attach(MIMEText(body))

    # send message
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_addr, password)
    server.sendmail(gmail_addr, to_addrs, msg.as_string())
    server.close()

if __name__ == '__main__':
    try:
        gmail_password = os.environ['gmail_password']
        email(from_addr='orenshk@gmail.com',
              to_addrs=['orenshk@gmail.com', 'oshklars@sfu.ca'],
              subject='testing email sender',
              body='yay!',
              username='orenshk@gmail.com',
              password=gmail_password,
              service='gmail')
    except KeyError:
        print('password needs to be available in environment variable')


