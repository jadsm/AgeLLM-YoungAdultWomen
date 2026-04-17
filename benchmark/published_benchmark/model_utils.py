import requests
from google import genai
from google.genai import types
import base64
from utils import parse_content, config
import os
import subprocess

def generate_gemini(content: str,model: str) -> str:
    client = genai.Client(vertexai=True,api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"))
    
    # load system instruction from file
    with open("system_instructions.txt", "r") as f:
      si_text1 = f.read()

    # load contents from file
    contents = parse_content(content)

    output = []
    for chunk in client.models.generate_content_stream(
      model = model,
      contents = contents,
      config = config(si_text1),
      ):
      if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
          continue
      output.append(chunk.text)
    
    return "".join(output)
  
def generate_claude_sonnet(content: str,model: str) -> str:
    """
    Generates a response from the Claude Sonnet model using the Vertex AI API.

    Args:
        content: The user's input content.

    Returns:
        The generated response from the model.
    """
    endpoint = "asia-southeast1-aiplatform.googleapis.com"
    location_id = "asia-southeast1"
    project_id = "imperial-410612"
    model_id = model
    method = "streamRawPredict"

    url = f"https://{endpoint}/v1/projects/{project_id}/locations/{location_id}/publishers/anthropic/models/{model_id}:{method}"

    # Get access token
    # This assumes gcloud CLI is installed and authenticated
    # You might need to implement a more robust way to get the token in a production environment
    
    access_token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode("utf-8").strip()

    with open("system_instructions.txt", "r") as f:
      si_text1 = f.read()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "stream": False,  # Set to False for a single response, True for streaming
        "max_tokens": 20000,
        "top_p": 0.95,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": si_text1+content}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Assuming the response is JSON and contains a 'text' field in the last message
    response_data = response.json()
    if "content" in response_data and response_data["content"]:
        return response_data["content"][0].get("text", "")
    return ""


def generate_openai(content: str,model: str,location_id:str = "global") -> str:
    """
    Generates a response from the models using the Vertex AI API.

    Args:
        content: The user's input content.
        model: The model to use for generation, e.g., "qwen/qwen3-next-80b-a3b-instruct-maas", "openai/gpt-oss-20b-maas", "meta/llama-3.3-70b-instruct-maas"
        location_id: The location of the model endpoint, e.g., "global", "us-central1". Default is "global".

    Returns:
        The generated response from the model.
    """

    endpoint = "aiplatform.googleapis.com" if location_id == "global" else f"{location_id}-aiplatform.googleapis.com"
    project_id = "imperial-410612"
    
    url = f"https://{endpoint}/v1beta1/projects/{project_id}/locations/{location_id}/endpoints/openapi/chat/completions" 
    # Get access token
    # This assumes gcloud CLI is installed and authenticated
    # You might need to implement a more robust way to get the token in a production environment
    
    access_token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode("utf-8").strip()

    with open("system_instructions.txt", "r") as f:
      si_text1 = f.read()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8" if location_id == "global" else "application/json"
    }

    payload = {
        "model": model,
        "stream": False,  # Set to False for a single response, True for streaming
        "max_tokens": 20000,
        "temperature": 0,
        "top_p": 0.95,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": si_text1+content}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Assuming the response is JSON and contains a 'text' field in the last message
    response_data = response.json()
    if "choices" in response_data and response_data["choices"]:
        return response_data["choices"][0].get("message", {}).get("content", "")
    return ""
