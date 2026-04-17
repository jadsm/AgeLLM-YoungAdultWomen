
import os
from model_utils import generate_claude_sonnet, generate_gemini, generate_openai
import json

with open("/Users/jdelgad1/Desktop/Juan/code/.creds/vertex_api.txt", "r") as f:
  os.environ["GOOGLE_CLOUD_API_KEY"] = f.read().strip()

def generate(content,model)-> str:
  if model.startswith("gemini"):
    return generate_gemini(content, model)
  elif model.startswith("claude"):
    return generate_claude_sonnet(content,model)
  elif model.startswith("qwen"):
    return generate_openai(content, "qwen/"+model)
  elif model.startswith("gpt"):
    return generate_openai(content, "openai/"+model)
  elif model.startswith("llama"):
    return generate_openai(content, "meta/"+model,location_id="us-central1")
  else:
    raise ValueError(f"Model {model} not supported")


if __name__ == "__main__":
  
  # load contents from file
  with open("data/source_data/health_info_example.jsonl", "r") as f:
    content_strings = [json.loads(line)["input"] for line in f.read().splitlines()]


  #  model = "gemini-3.1-pro-preview","gemini-3.1-flash-lite-preview","claude-sonnet-4-6",
  # "qwen3-next-80b-a3b-instruct-maas","gpt-oss-20b-maas"
  # models = ["llama-3.3-70b-instruct-maas"]
  models = ["llama-3.2-90b-vision-instruct-maas"]
  
  for model in models:
    responses = [generate(content, model) for content in content_strings]
    with open(f"outputs/output_{model}.txt", "w") as f:
      for response in responses:
        f.write(response + "\n")
