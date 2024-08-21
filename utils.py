import os
import json
import time
import yaml
import random
import requests

from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs

import re

def remove_special_tokens(messages):
    # List of special tokens to remove
    special_tokens = [
        r'<\|im_start\|>', r'<\|im_end\|>',  # ChatML tokens
        r'<s>', r'</s>', r'<\|eot_id\|>',     # LLaMA tokens
        r'<\|endoftext\|>',                  # GPT special token
    ]

    new_messages = []
    
    # Combine all special tokens into a single regex pattern
    pattern = '|'.join(special_tokens)
    
    for m in messages:


        # Remove all instances of special tokens from the prompt
        m['content'] = re.sub(pattern, '', m['content'])
    
        # Remove any extra whitespace that might have been left behind
        m['content'] = ' '.join(m['content'].split())
        new_messages.append(m)
    
    return new_messages

def remove_duplicate_char_ngrams(messages, n):
    new_messages = []
    for m in messages:
        # Generate character n-grams
        text = m['content']
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        
        # Keep track of seen n-grams
        seen = set()
        result = []
        
        for i, ngram in enumerate(ngrams):
            if ngram not in seen:
                # This is a new n-gram, add it to the result
                result.append(text[i])
                seen.add(ngram)
            else:
                # This is a duplicate n-gram, skip it
                pass
        
        # Add any remaining characters
        result.extend(text[len(text)-n+1:])
        
        # Join the characters back into a string
        s = ''.join(result)
        m['content'] = s
        new_messages.append(m)
    return new_messages

def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e, dir(e))
            #NOTE: this has the potential to produce some pretty weird strings; should be careful here
            # if e.error['code'] == 'invalid_prompt':
            messages = remove_special_tokens(messages)
            messages = remove_duplicate_char_ngrams(messages, 50)

        except KeyError:
            print(type(e), e)
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output

def chat_completion_huggingface(model, conv, temperature, max_tokens):
    API_BASE = "https://api-inference.huggingface.co/models/"
    API_URL = API_BASE + model
    headers = {"Authorization": "Bearer " + str(os.environ["HUGGINGFACE_API_KEY"])}
    query = {
        "inputs": conv,
    }
    output = requests.post(API_URL, headers=headers, json=query)
    output = output.json()
    print(output)
    breakpoint()
    return 

def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts":[
                        {"text": message}
                    ]
                }],
                "safetySettings": safety_settings,
                "generationConfig":{
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output
    


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

import json
import re

def write_with_subscores(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            
            for game in data['games']:
                judgment = game['judgment']
                
                # Define regex patterns
                patterns = {
                    'correctness_score': r'Correctness: \(\(([AB<>=]+)\)\)',
                    'completeness_score': r'Completeness: \(\(([AB<>=]+)\)\)',
                    'safety_score': r'Safety: \(\(([AB<>=]+)\)\)',
                    'conciseness_score': r'Conciseness: \(\(([AB<>=]+)\)\)',
                    'style_score': r'Style: \(\(([AB<>=]+)\)\)'
                }
                
                # Extract scores using regex
                for key, pattern in patterns.items():
                    match = re.search(pattern, judgment)
                    game[key] = match.group(1) if match else ''
                
                # Ranking Logic
                game['score_final'] = game['style_score']
                # Reorder the dictionary to insert new keys after 'score'
                keys = list(game.keys())
                score_index = keys.index('score')
                new_keys = keys[:score_index+1] + list(patterns.keys()) + keys[score_index+1:]
                new_keys = [k for k in new_keys if k not in patterns]
                
                game = {k: game[k] for k in new_keys}
            
            # Write the modified data to the output file
            json.dump(data, outfile)
            outfile.write('\n')