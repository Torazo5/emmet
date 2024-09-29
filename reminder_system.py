# reminder_system.py
import datetime
import json
import os
import threading
import time
import subprocess
from shared_state import shared_state

class ReminderSystem:
    def __init__(self, client, audio_function):
        self.client = client
        self.audio = audio_function

    def compute_exact_datetime_sub_gpt(self, current_time, user_request):
        prompt = (
            f"The current date and time is {current_time}. "
            f"Given the following user request: '{user_request}', "
            f"compute the exact date and time for the reminder in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). "
            f"Only provide the datetime in ISO 8601 format, do not include any extra text."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=20,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        datetime_str = response.choices[0].message.content.strip()
        try:
            datetime.datetime.fromisoformat(datetime_str)
            return datetime_str
        except ValueError:
            return None

    def check_reminders(self):
        while True:
            now = datetime.datetime.now()
            reminders_to_keep = []
            if os.path.exists('reminders.json'):
                with open('reminders.json', 'r') as f:
                    reminders = json.load(f)
                for reminder in reminders:
                    reminder_time = datetime.datetime.fromisoformat(reminder['reminder_time'])
                    if now >= reminder_time:
                        self.timer_finished(reminder['message'])
                    else:
                        reminders_to_keep.append(reminder)
                with open('reminders.json', 'w') as f:
                    json.dump(reminders_to_keep, f, indent=4)
            time.sleep(30)

    def process_reminder_request(self, user_request):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        exact_datetime = self.compute_exact_datetime_sub_gpt(current_time, user_request)
        if exact_datetime is None:
            return "Sorry, I couldn't understand the reminder time."
        reminder = {
            'reminder_time': exact_datetime,
            'message': user_request
        }
        reminders = []
        if os.path.exists('reminders.json'):
            with open('reminders.json', 'r') as f:
                reminders = json.load(f)
        reminders.append(reminder)
        with open('reminders.json', 'w') as f:
            json.dump(reminders, f, indent=4)
        return f"Reminder set for {exact_datetime}: {user_request}"

    def timer_finished(self, message):
        while shared_state.is_speaking_or_listening:
            time.sleep(1)
        shared_state.is_speaking_or_listening = True
        subprocess.run(['afplay', 'rg1.mp3'])
        time.sleep(1)
        if message:
            print(message)
            self.audio(message)
        else:
            print("Time's up!")
            self.audio("Time's up!")
        time.sleep(0.5)
        subprocess.run(['afplay', 'rg1-over.mp3'])
        shared_state.is_speaking_or_listening = False

    def set_timer(self, duration_seconds, message=""):
        print(f"Timer set for {duration_seconds} seconds.")
        timer_thread = threading.Timer(duration_seconds, self.timer_finished, [message])
        timer_thread.start()
        return f"Timer set for {duration_seconds} seconds."
