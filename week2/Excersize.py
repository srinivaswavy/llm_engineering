import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import json
import requests

load_dotenv()

APPOINTMENT_API_URL = "http://localhost:8801"

def get_doctors():
    """Get a list of all doctors."""
    resp = requests.get(f"{APPOINTMENT_API_URL}/doctors")
    resp.raise_for_status()
    return resp.json()


def get_available_slots(doctor_id: str, date: str):
    """Get available slots for a doctor on a specific date (YYYY-MM-DD)."""
    resp = requests.get(f"{APPOINTMENT_API_URL}/slots/{doctor_id}", params={"date": date})
    resp.raise_for_status()
    return resp.json()


def book_appointment(patient_id:str, doctor_id: str, appointment_date: str):
    """Book an appointment for a patient with a doctor at a specific datetime (ISO format)."""
    payload = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "appointment_date": appointment_date
    }
    resp = requests.post(f"{APPOINTMENT_API_URL}/appointments", json=payload)
    resp.raise_for_status()
    return resp.json()



get_doctors_function = {
    "name": "get_doctors",
    "description": """Get list of all doctors. Call this function whenever customer is looking for list of doctors they can choose for consultation. 
    If the function call is failed for any reason, respond that some technical error occured.
    """,
    "parameters": {
        "type": "object",
        "properties": {
        },
        "required": [],
        "additionalProperties": False
    }
}

get_slots_function = {
    "name": "check_doctor_availability",
    "description": "Checks if the doctor is available on a specific date (YYYY-MM-DD). If date is not supplied in YYYY-MM-DD format, guide the user. If the function call is failed for any reason, respond that some technical error occured.",
    "parameters": {
        "type": "object",
        "properties": {
            "doctor_id": {
                "type": "string",
                "description": "Id of the doctor the user wants to see if available on a date.",
            },
            "date": {
                "type": "string",
                "description": "Date on which user wants to see if the particular doctor is available",
            },
        },
        "required": ["doctor_id","date"],
        "additionalProperties": False
    }
}

book_appointment_function = {
    "name": "book_appointment",
    "description": "Book an appointment for a patient with a doctor at a specific datetime (ISO format). Ask the patient id,doctor id and date of appointment. If date is not supplied in YYYY-MM-DD format, guide the user. If the function call is failed for any reason, respond that some technical error occured.",
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "Id of the patient",
            },
            "doctor_id": {
                "type": "string",
                "description": "Id of the doctor.",
            },
            "date": {
                "type": "string",
                "description": "Date on which user wants to see the doctor",
            },
        },
        "required": ["doctor_id","date"],
        "additionalProperties": False
    }
}

tools = [{"type":"function","function":get_doctors_function},
         {"type":"function","function":get_slots_function},
         {"type":"function","function":book_appointment_function}]


MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA_3_2 = 'llama3.2'
MODEL_LLAMA_3 = 'llama3'
MODEL_qwen = 'qwen3:32b'
MODEL_QWEN_8b = 'qwen3:8b'
MODEL_QWEN_14b = 'qwen3:14b'
MODEL_LLAMA_3_1 = 'llama3.1:8b'
MODEL_GRANITE = 'granite3.1-dense'
MODEL_GEMMA3_12b = 'gemma3:12b'
def get_open_ai(llm_url):
    if(llm_url):
        return OpenAI(base_url=llm_url,api_key='ollama')
    else:
        return OpenAI()

#openai = get_open_ai(llm_url="http://localhost:11434/v1")
openai = get_open_ai(None)

system_message = "You are an AI assistance to help users to book a doctor appointment. "
system_message += "Present the available doctors to the patient."
system_message += "On selection of doctor id, ask for date of appointment in YYYY-MM-DD format. If date given in any other format, ask them to correct the format."
system_message += "Check if the doctor is available on that particular date. Based on check_doctor_availability tool response, if the doctor is not available on the selected date, ask to choose another date or doctor."
system_message += "If doctor is available on the date, ask for patient id for booking the appointment. However, you need just the date to book the appointment."
system_message += "Use the selected doctor id, date and given patient id to book the appointment. Use book_appointment tool to book the appointment. Verify if the tool was called to book the appointment."
system_message += "If there are any function calls failed in the process, tell the user that there is some technical error."
system_message += "If everything goes well, confirm the user that the appointment is booked."
system_message += "Always, present the doctors and doctor availability data provided by the tools. Do not try to present any doctors or doctor availability data other than what you are given by tools" 

import re

def strip_think_blocks(text):
    # Removes anything inside <think>...</think> (including the tags)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def chat_with_gpt40(audio, text, history):
    MODEL = MODEL_GPT
    # 1. Get user input from either audio or text
    user_text = None
    if audio is not None:
        with open(audio, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_text = transcript.text
    elif text and text.strip():
        user_text = text.strip()
    else:
        return None, "Please provide a voice or text input.", history if history else [], []

    # 2. Prepare conversation history for GPT (exclude system messages from chat display)
    messages = []
    if history:
        for msg in history:
            role = getattr(msg, "role", None) if hasattr(msg, "role") else msg.get("role")
            if role != "system":
                messages.append(msg)
    messages = [{"role": "system", "content": system_message}] + messages
    messages = messages + [{"role": "user", "content": user_text}]

    # 3. Send text to GPT-4o-mini
    chat_response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )

    while chat_response.choices[0].finish_reason == "tool_calls":
        message = chat_response.choices[0].message
        print('message', message)
        tool_response = handle_tool_call(message)
        print('tool_response',tool_response)
        messages.append(message)
        messages.append(tool_response)
        chat_response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    gpt_text = chat_response.choices[0].message.content
    print('gpt_text', gpt_text)
    gpt_text = strip_think_blocks(gpt_text)
    messages.append({"role": "assistant", "content": gpt_text})

    # 4. Convert GPT response to speech
    tts_response = openai.audio.speech.create(
        model="tts-1",
        input=gpt_text,
        voice="alloy"
    )
    out_path = "gpt_response.mp3"
    with open(out_path, "wb") as f:
        f.write(tts_response.content)

    # Simplified chat history for display: only user/assistant messages, skip system/tool
    chat_pairs = []
    display_msgs = [
        m if isinstance(m, dict) else m.__dict__
        for m in messages
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) in ("user", "assistant")
    ]
    last_role = None
    last_content = ""
    for m in display_msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            # If previous was also user, append as new turn
            if last_role == "user":
                chat_pairs.append((last_content, ""))
            last_content = content
            last_role = "user"
        elif role == "assistant":
            # If previous was user, pair them
            if last_role == "user":
                chat_pairs.append((last_content, content))
                last_content = ""
                last_role = None
            else:
                # Assistant message without user, show as (None, assistant)
                chat_pairs.append(("", content))
                last_content = ""
                last_role = None
    # If last message was user and not paired
    if last_role == "user":
        chat_pairs.append((last_content, ""))

    return out_path, gpt_text, messages, chat_pairs


def handle_tool_call(message):
    print('tool_message',message)
    tool_call = message.tool_calls[0]
    if(tool_call.function.name == "get_doctors"):
        result = get_doctors()
        return {
            "role": "tool",
            "content":  json.dumps(result),
            "tool_call_id": tool_call.id
        }
    elif(tool_call.function.name == "check_doctor_availability"):
        arguments = json.loads(tool_call.function.arguments)
        doctor_id = arguments.get('doctor_id')
        date = arguments.get('date')
        result = get_available_slots(doctor_id,date)
        return {
            "role": "tool",
            "content":  json.dumps(result),
            "tool_call_id": tool_call.id
        }
    elif(tool_call.function.name == "book_appointment"):
        arguments = json.loads(tool_call.function.arguments)
        patient_id = arguments.get('patient_id')
        doctor_id = arguments.get('doctor_id')
        date = arguments.get('date')
        result = book_appointment(patient_id,doctor_id,date)
        return {
            "role": "tool",
            "content":  json.dumps(result),
            "tool_call_id": tool_call.id
        }
        
    return {
            "role": "tool",
            "content": "I didn't get you.",
            "tool_call_id": tool_call.id
        }

iface = gr.Interface(
    fn=chat_with_gpt40,
    inputs=[
        gr.Audio(type="filepath", show_label=True, show_download_button=False, interactive=True),
        gr.Textbox(label="Type your message (or use voice above)", lines=2, interactive=True),
        gr.State([])  # history state
    ],
    outputs=[
        gr.Audio(label="GPT-4o-mini Voice Response", type="filepath", autoplay=True),
        gr.Textbox(label="GPT-4o-mini Text Response"),
        gr.State(),  # updated history
        gr.Chatbot(label="Conversation History")
    ],
    title="Voice/Text Chat with GPT-4o-mini",
    description="Speak or type to GPT-4o-mini and hear its response. Conversation history is shown below."
)

if __name__ == "__main__":
    iface.launch()
