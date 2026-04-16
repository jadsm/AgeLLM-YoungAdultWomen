import requests
from google import genai
from google.genai import types
import base64
from utils import parse_content, config
import os

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
    import subprocess
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
        "max_tokens": 512,
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
    if "messages" in response_data and response_data["messages"]:
        last_message = response_data["messages"][-1]
        if "content" in last_message and last_message["content"]:
            return last_message["content"][0].get("text", "")
    return ""

#         cat << EOF > request.json
# {
#     "anthropic_version": "vertex-2023-10-16"
#     ,"stream": true
#     ,"max_tokens": 512
#     ,"top_p": 0.95
#     ,"messages": [
#         {
#             "role": "user",
#             "content": [
#             ]
#         }
#     ]
# }
# EOF

# ENDPOINT="asia-southeast1-aiplatform.googleapis.com"
# LOCATION_ID="asia-southeast1"
# PROJECT_ID="imperial-410612"
# MODEL_ID="claude-sonnet-4-6"
# METHOD="streamRawPredict"

# curl -X POST \
#   -H "Authorization: Bearer $(gcloud auth print-access-token)" \
#   -H "Content-Type: application/json; charset=utf-8" \
#   -d @request.json \
# "https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION_ID}/publishers/anthropic/models/${MODEL_ID}:${METHOD}"