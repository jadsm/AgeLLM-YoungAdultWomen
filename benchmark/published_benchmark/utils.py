from google.genai import types

def parse_content(txt_list: list[str]) -> list[types.Content]:
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=txt) for txt in txt_list
        ]
        ),
    ]
    return contents

def config(si_text1) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
    temperature = 0,
    top_p = 0.95,
    max_output_tokens = 20000,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    response_mime_type = "application/json",
    system_instruction=[types.Part.from_text(text=si_text1)],
    thinking_config=types.ThinkingConfig(
      thinking_level="LOW",
    ),
  )
  