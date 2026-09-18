
import os
from model_utils import generate_claude_sonnet, generate_gemini, generate_openai
import json
import time
HERE = os.path.dirname(os.path.abspath(__file__))

with open("/Users/jdelgad1/Desktop/Juan/code/.creds/vertex_api.txt", "r") as f:
  os.environ["GOOGLE_CLOUD_API_KEY"] = f.read().strip()

def generate(content,model, si_text1, seed=None)-> str:
  if model.startswith("gemini"):
    return generate_gemini(content, model, si_text1, seed=seed)
  elif model.startswith("claude"):
    return generate_claude_sonnet(content,model, si_text1)
  elif model.startswith("qwen"):
    return generate_openai(content, "qwen/"+model, si_text1, seed=seed)
  elif model.startswith("gpt"):
    return generate_openai(content, "openai/"+model, si_text1, seed=seed)
  elif model.startswith("minimax"):
    return generate_openai(content, "minimaxai/"+model, si_text1, seed=seed)
  elif model.startswith("llama"):
    return generate_openai(content, "meta/"+model,si_text1,location_id="us-central1",  seed=seed)
  else:
    raise ValueError(f"Model {model} not supported")


if __name__ == "__main__":
  # paths = {"women_pre_natal":os.path.join(HERE, "data/source_data/women_before_pregnancy.jsonl"),
  #          "women_post_natal":os.path.join(HERE, "data/source_data/women_after_pregnancy.jsonl")}
  paths = {
            "8yo":os.path.join(HERE, "data/source_data/gusto_y8_selected20.jsonl"),
             "10yo":os.path.join(HERE, "data/source_data/gusto_y10_selected20.jsonl"),
             "12yo":os.path.join(HERE, "data/source_data/gusto_y12_selected20.jsonl")
             }

  # prompts = [os.path.join(HERE, 'system_instructions/system_instructions.txt'),
  #             os.path.join(HERE, 'system_instructions/system_instructions_simple.txt'),
  #             os.path.join(HERE, 'system_instructions/system_instructions_supersimple.txt')]

  prompts = [os.path.join(HERE, 'system_instructions/system_instructions.txt')]
  # "gemini-3.1-pro-preview","gemini-3.8-flash","gemini-3.5-flash-lite","claude-sonnet-5",
  models = ["minimax-m2-maas"]#"qwen3-235b-a22b-instruct-2507-maas","gpt-oss-120b-maas",
  
  # repeats = 5
  # for k in [0]:#range(repeats):
  for k,prompt in enumerate(prompts):
    for name, path in paths.items():
      # load contents from file
      with open(path, "r") as f:
        content_strings = [json.loads(line)["input"] for line in f.read().splitlines()]
      
      # load system instruction from file
      with open(prompt, "r") as f:
        si_text1 = f.read()
      
      for model in models:
        # responses = [generate(content, model, seed=k) for content in content_strings]
        responses = [generate(content, model, si_text1) for content in content_strings]
        # with open(f"outputs/output_{name}_{model}_{k}_seed{k}.txt", "w") as f:
        with open(os.path.join(HERE, f"outputs/output_{name}_{model}_prompt{k}.txt"), "w") as f:
          for response in responses:
            f.write(response + "\n")
        print(f'Finished generating! {name}_{model}_prompt{k}')
        # wait 10 seconds to give some space to the API and not exhaust them
        time.sleep(10)
