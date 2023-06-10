import os
import openai
import csv
import sys
import hashlib
import pickle
import json
import argparse
import time
from tqdm import tqdm
import random
import numpy as np
from config import config

random.seed(1234)

# create a config.py file with the dictionary config = {"OPENAI_API_KEY": "xxxx"}
OPENAI_API_KEY = config["OPENAI_API_KEY"]

class OpenAICommunicator():

    def __init__(self, options):
        data_path = options["data_path"]
        openai.api_key = OPENAI_API_KEY
        # load main test set
        with open(options["data_path"], 'r') as fp:
            self.data = json.load(fp)
        self.max_tokens = options["max_tokens"]
        self.cache_path = options["cache_path"]
        self.save_path = options["save_path"]
        self.cached_responses = self.load_cache_if_exists()

    def load_cache_if_exists(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as handle:
                cache_file = pickle.load(handle)
                return cache_file
        else:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            return {}

    def make_openai_api_call(self, prompt):
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=self.max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return self.parse_api_response(response)
        except openai.error.ServiceUnavailableError:
            print("Service unavailable error hit")
            time.sleep(20)
            return self.make_openai_api_call(prompt)
        except openai.error.RateLimitError as e:
            if 'You exceeded your current quota' in e.user_message:
                print('Key exhausted; use different key and re-run')
                sys.exit()
            else:
                print("Server overloaded; rate limit hit")
                time.sleep(20)
            return self.make_openai_api_call(prompt)

    def parse_api_response(self, response):
        choices = response["choices"]
        return choices[0]["text"].strip(), response

    def run_inference(self):
        answers = {}
        for user_id, user_data in tqdm(self.data.items()):
            prompt = user_data["prompt"]
            hashed_prompt = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if hashed_prompt in self.cached_responses:
                response_text = self.cached_responses[hashed_prompt]['text']
            else:
                response_text, response = self.make_openai_api_call(prompt)
                self.cached_responses[hashed_prompt] = {'text': response_text, 'object': response}
                with open(self.cache_path, 'wb') as handle:
                    pickle.dump(self.cached_responses, handle)
                time.sleep(5)
            answers[user_id] = response_text

        out_fp = open(self.save_path, 'w+')
        fieldnames = ['user_id', 'prompt', 'prediction']
        writer = csv.DictWriter(out_fp, fieldnames, lineterminator='\n', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for user_id, user_data in tqdm(self.data.items()):
            info = {}
            info['user_id'] = user_id
            info['prompt'] = user_data["prompt"]
            info['prediction'] = answers[user_id]
            writer.writerow(info)
        out_fp.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Data JSON file path")
    parser.add_argument("--cache_path", help="Path with file to save GPT3 responses", default="../data/gpt3_cache/cache.pkl")
    parser.add_argument("--save_path", help="Path to save model predictions")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max tokens for GPT-3 to generate")
    args, _ = parser.parse_known_args()
    return args

def main(args):

    options = {}
    options["data_path"] = args.data_path
    options["cache_path"] = args.cache_path
    options["save_path"] = args.save_path
    options["max_tokens"] = args.max_tokens

    openai_communicator = OpenAICommunicator(options)
    openai_communicator.run_inference()

if __name__ == '__main__':
    args = parse_args()
    main(args)
