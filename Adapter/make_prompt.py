import json
import os

def create_prompt_json(source_dir, output_file=r'.\prompt.json'):
    prompts = []
    for filename in os.listdir(source_dir):
        prompt = {
            'source': rf".\thz_dataset\train\{filename}",
            'target': rf".\thz_dataset\target\{filename}",
            'prompt': "a scan with a defect" if "AMITIE_Tepex_WRC1331" in filename else "a scan without a defect"
        }
        prompts.append(prompt)
    print(prompts)
    with open(output_file, 'wt') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')

create_prompt_json(r"D:\700GHz\mat_files\train")
