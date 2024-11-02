import requests

def send_notification(topic:str,data:str):
    requests.post("https://ntfy.sh/" + topic,
    data=data.encode(encoding='utf-8'))

