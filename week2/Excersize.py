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

def chat_with_gpt40(audio, history):
    MODEL = MODEL_GPT 
    # 1. Transcribe audio to text
    with open(audio, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_text = transcript.text

    # 2. Prepare conversation history for GPT (exclude system messages from chat history)
    messages = []
    if history:
        # Filter out system messages from history
        for msg in history:
            if hasattr(msg, "role"):
                if msg.role != "system":
                    messages.append(msg)
            elif isinstance(msg, dict):
                if msg.get("role") != "system":
                    messages.append(msg)
    # Add system message for the current turn
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
    # Strip <think> blocks from the GPT response
    gpt_text = strip_think_blocks(gpt_text)
    messages.append({"role": "assistant", "content": gpt_text})

    # 4. Convert GPT response to speech
    tts_response = openai.audio.speech.create(
        model="tts-1",
        input=gpt_text,
        voice="alloy"
    )
    # Save audio to a temporary file
    out_path = "gpt_response.mp3"
    with open(out_path, "wb") as f:
        f.write(tts_response.content)

    # Prepare chat history for gr.Chatbot (exclude system messages)
    chat_pairs = []
    filtered = []
    for m in messages:
        if hasattr(m, "role"):
            if m.role != "system":
                filtered.append(m)
        elif isinstance(m, dict):
            if m.get("role") != "system":
                filtered.append(m)
    for i in range(0, len(filtered)-1, 2):
        user_msg = filtered[i].content if hasattr(filtered[i], "content") else filtered[i].get("content", "")
        if hasattr(filtered[i+1], "role"):
            role = filtered[i+1].role
            assistant_msg = filtered[i+1].content if role == "assistant" and hasattr(filtered[i+1], "content") else ""
        else:
            assistant_msg = filtered[i+1].get("content", "") if filtered[i+1].get("role", "") == "assistant" else ""
        chat_pairs.append((user_msg, assistant_msg))
    if len(filtered) % 2 == 1 and (hasattr(filtered[-1], "role") and filtered[-1].role == "user" or isinstance(filtered[-1], dict) and filtered[-1].get("role") == "user"):
        last_msg = filtered[-1].content if hasattr(filtered[-1], "content") else filtered[-1].get("content", "")
        chat_pairs.append((last_msg, ""))

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
        gr.State([])  # history state
    ],
    outputs=[
        gr.Audio(label="GPT-4o-mini Voice Response", type="filepath", autoplay=True),
        gr.Textbox(label="GPT-4o-mini Text Response"),
        gr.State(),  # updated history
        gr.Chatbot(label="Conversation History")
    ],
    title="Voice Chat with GPT-4o-mini",
    description="Speak to GPT-4o-mini and hear its response. Conversation history is shown below."
)

if __name__ == "__main__":
    iface.launch()
