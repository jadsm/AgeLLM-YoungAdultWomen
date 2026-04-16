
import os
from model_utils import generate_claude_sonnet, generate_gemini
import json

with open("/Users/jdelgad1/Desktop/Juan/code/.creds/vertex_api.txt", "r") as f:
  os.environ["GOOGLE_CLOUD_API_KEY"] = f.read().strip()

def generate(content,model)-> str:
  if model.startswith("gemini"):
    return generate_gemini(content, model)
  elif model.startswith("claude"):
    return generate_claude_sonnet(content,model)

if __name__ == "__main__":
  
  # load contents from file
  with open("data/source_data/health_info_example.jsonl", "r") as f:
    content_strings = [json.loads(line)["input"] for line in f.read().splitlines()]


  #  model = "gemini-3.1-pro-preview","gemini-3.1-flash-lite-preview",
  models = ["claude-sonnet-4-6"]
  for model in models:
    responses = [generate(content, model) for content in content_strings]
    with open(f"outputs/output_{model}.txt", "w") as f:
      for response in responses:
        f.write(response + "\n")


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



# cat << EOF > request.json
# {
#     "model": "qwen/qwen3-next-80b-a3b-instruct-maas"
#     ,"stream": true
#     ,"max_tokens": 8192
#     ,"temperature": 0
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

# ENDPOINT="aiplatform.googleapis.com"
# REGION="global"
# PROJECT_ID="imperial-410612"

# curl \
# -X POST \
# -H "Content-Type: application/json" \
# -H "Authorization: Bearer $(gcloud auth print-access-token)" \
# "https://${ENDPOINT}/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions" -d '@request.json'




# cat << EOF > request.json
# {
#     "model": "openai/gpt-oss-20b-maas"
#     ,"stream": true
#     ,"max_tokens": 4096
#     ,"temperature": 0
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

# ENDPOINT="aiplatform.googleapis.com"
# REGION="global"
# PROJECT_ID="imperial-410612"

# curl \
# -X POST \
# -H "Content-Type: application/json" \
# -H "Authorization: Bearer $(gcloud auth print-access-token)" \
# "https://${ENDPOINT}/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions" -d '@request.json'