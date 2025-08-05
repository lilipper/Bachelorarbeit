import json
import os
import pandas as pd

def create_prompt_json(source_dir):
    prompts = []
    data=[]
    for filename in os.listdir(source_dir):
        prompt = {
            'source': rf".\thz_dataset\train\{filename}",
            'target': rf".\thz_dataset\target\{filename}",
            'prompt': "a scan with a defect" if "AMITIE_Tepex_WRC1331" in filename else "a scan without a defect"
        }
        data.append({"filename": filename, "label": "defect" if "AMITIE_Tepex_WRC1331" in filename else "healthy"})
        prompts.append(prompt)
    with open(fr'.\{os.path.basename(source_dir)}_prompt.json', 'wt') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    df = pd.DataFrame(data)
    df.to_csv(fr'.\{os.path.basename(source_dir)}_labels.csv', index=False)
    df = pd.DataFrame(prompts)
    df.to_csv(fr'.\{os.path.basename(source_dir)}_prompt.csv', index=False)

create_prompt_json(r"D:\700GHz\mat_files\train")
create_prompt_json(r"D:\700GHz\mat_files\test")
