from tqdm.auto import tqdm
from openai import AzureOpenAI, BadRequestError
import concurrent.futures
from tqdm.auto import tqdm
import os
import json
import argparse
import pandas as pd
from glob import glob
import re

def make_sys_message(l, c):
    example = """{
    "truthfulness": 10,
    "helpfulness": 10,
    "succinctness": 10,
    "importance": 10,
    "fluency": 10,
    "naturalness": 10,
}"""
    return f"""Given a prompt and response in {l}, evaluate the response from 1-10 on the following criteria:
    
Truthfulness - Is the response free from false statements and facts?
Helpfulness - How well does the response reply to the original prompt?
Succinctness - Is the response able to reply to the prompt without extraneous information?
Importance - Does the response concern a topic that is generally considered important?
Fluency - Does the response respond in fluent, natural {l}?

Also evaluate the prompt on the following criteria:

Naturalness - Is the prompt a natural {c} style prompt in {l}?

Be harsh in your evaluation.

Reply only with a JSON fomatted like so:

{example}

without any Markdown formatting (i.e. ```) around it."""

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2024-04-01-preview",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
)

def get_openai_response(prompt, response, language, prompt_type, max_tokens = 128):

    response = client.chat.completions.create(
        model="gpt4o-2024-05-13",
        messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": make_sys_message(language, prompt_type)
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "<<PROMPT>>\n\n" + prompt + "\n\n<<RESPONSE>>\n\n" + response
            }
          ]
        }
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    choice = response.choices[0]
        
    return {"finish_reason": choice.finish_reason, "content": choice.message.content}

def parse_jiten(raw_text):
    pattern = r"<<PROMPT>>([\s\S]+)<<RESPONSE>>([\s\S]+)"

    match = re.search(pattern, raw_text)
    if match:
        prompt = match.group(1).strip()
        response = match.group(2).strip()
        return prompt, response
    else:
        return None, None

def handle_openai_call(input_data_item):
    prompt_response, language, prompt_type = input_data_item
    if prompt_response is None or not isinstance(prompt_response["content"], str):
        return {"error": "Null prompt response"}
    
    prompt, response = parse_jiten(prompt_response["content"])
    
    if prompt is None or response is None:
        return {"error": "Failed prompt/response parse"}

    try:
        return get_openai_response(prompt, response, language, prompt_type)
    except BadRequestError as bre:
        print(json.loads(bre.response.content))
        return {"bad_request": str(json.loads(bre.response.content)["error"]["innererror"])}
    except Exception as e:
        print(e)
        return {"error": str(e)}

def run_generation():
    paths = [x for x in sorted(glob("./gen_data/*.parquet")) if "selected" not in x]
    
    chunk_size = 1_000

    for path in tqdm(paths):
        df = pd.read_parquet(path)

        input_data = list(zip(
            df["generated_prompt_response"].tolist(),
            df["language_name"].tolist(),
            df["prompt_type"].tolist(),
        ))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            response_results = list(tqdm(executor.map(handle_openai_call, input_data), total=len(input_data)))

        df["filter_response"] = [handle_openai_call(x) for x in input_data]

        df.to_parquet(f"./filtered_data/{os.path.basename(path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--num_samples_per_cluster', type=int, required=False, default=1)

    # Parse the arguments
    args = parser.parse_args()

    run_generation()
