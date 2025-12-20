import argparse

import requests
import json
import time

def chat_request(url, data, headers, auth):
    while True:
        user_input = input("Enter your message: ")
        # use user_input if message is empty
        data["messages"].append({
            "role": "user",
            "content": user_input,
            "timestamp": int(time.time() * 1000)
        })

        print("Loading...")
        response = requests.post(url, headers=headers, data=json.dumps(data), auth=auth)
        print(response.status_code)

        if response.status_code == 200:
            response_data = response.text
            assistant_message = response_data
            print("Assistant: ", assistant_message)
            data["messages"].append({
                "role": "assistant",
                "content": assistant_message,
                "timestamp": int(time.time() * 1000)
            })
        else:
            print("Error: ", response.text)
        
        # catch KeyboardInterrupt to exit the loop
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

def gpt_request(session_id, message=None, prompt=None, temperature=0.3, 
                model_id="gpt-4o", model_name="gpt-4o", max_length=24000, token_limit=8000, chatmode=False):
    url = "https://ki-chat.sikt.no/api/chat"

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/json",
        "Cookie": f"session={session_id}",
        "Origin": "https://ki-chat.sikt.no",
        "Referer": "https://ki-chat.sikt.no/",
        "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    }
    default_sys_prompt = "Follow the user's instructions attentively. Use markdown in responses only when it enhances clarity or presentation."
    prompt = prompt if prompt else default_sys_prompt
    data = {
        "model": {
            "id": model_id,
            "name": model_name,
            # "maxLength": 24000,
            # "tokenLimit": 8000
            "maxLength": max_length,
            "tokenLimit": token_limit
        },
        "messages": [],
        "key": "",
        "prompt": prompt,
        "temperature": temperature,
        "lang": "default"
    }

    # Replace 'username' and 'password' with your actual username and password
    auth = ('username', 'password')

    if chatmode:
        chat_request(url, data, headers, auth)
    else:
        data["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": int(time.time() * 1000)
        })

        response = requests.post(url, headers=headers, data=json.dumps(data), auth=auth)
        print(response.status_code)

        if response.status_code == 200:
            response_data = response.text
            assistant_message = response_data
            print("Assistant: ", assistant_message)
            data["messages"].append({
                "role": "assistant",
                "content": assistant_message,
                "timestamp": int(time.time() * 1000)
            })
        else:
            print("Error: ", response.text)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str, help="Session ID")
    args = parser.parse_args()
    session_id = args.session_id
    
    gpt_request(session_id, chatmode=True)

