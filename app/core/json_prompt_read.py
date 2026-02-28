import json

#read json file and return the prompt
def read_json_prompt(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['prompt']




