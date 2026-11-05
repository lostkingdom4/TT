from openai import OpenAI
import json
import sys
from dotenv import load_dotenv
import os

def check_with_dp(prompt, model="deepseek-reasoner", system_prompt=None, stream=True):
    """
    Sends a chat prompt to the DeepSeek API and returns the response.
    Shows the streaming output as it's generated.

    :param prompt: The message to send to Ollama.
    :param model: The model to use for generation.
    :return: The complete response from Ollama.
    """
    print("Streaming response:")
    full_response = {"reasoning": "", "response": ""}
    load_dotenv()
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    if system_prompt == None:
        system_prompt = "You are a professional code reviewer."
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=stream,
            temperature=0.0,
        )

    else:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=stream,
            temperature=0.0,
            response_format={
            'type': 'json_object'
        }
        )

    if stream:
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                # Handle reasoning content if available (some models may have this)
                reasoning_text = getattr(delta, 'reasoning_content', None)
                if reasoning_text:
                    # print(reasoning_text, end='', flush=True)
                    full_response["reasoning"] += reasoning_text

                # Handle regular content
                chunk_content = delta.content
                if chunk_content:
                    print(chunk_content, end='', flush=True)
                    full_response["response"] += chunk_content
    else:
        response_text = json.loads(response.choices[0].message.content)
        full_response = {
            "reasoning": response.choices[0].message.reasoning_content,
            "response": response_text
        }

    print("\n\nEnd of streaming response")
    return full_response