import json
import os
import pandas as pd

def create_prompt_json(source_dir):
    prompts = []
    data=[]
    for filename in os.listdir(source_dir):
        prompt = {
            'source': rf"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train/{filename}",
            'target': rf"rf/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/target/{filename.replace('.mat', '.tiff')}",
            'prompt': "a scan with a defect" if "AMITIE_Tepex_WRC1331" in filename else "a scan without a defect"
        }
        data.append({"filename": filename, "label": "defect" if "AMITIE_Tepex_WRC1331" in filename else "healthy"})
        prompts.append(prompt)
    json_data = {"prompts":prompts}
    with open(fr'{os.path.basename(source_dir)}_prompt_image_per_mat.json', 'wt') as f:
        f.write(json.dumps(json_data, indent=4))

    prompts = []
    data=[]
    for filename in os.listdir(source_dir):
        prompt = {
            'source': rf"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train/{filename}",
            'target': rf"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/target/image_z_AMITIE_Tepex_104-RG600_750GHz_0004_N700.tiff",
            'prompt': "a scan with a defect" if "AMITIE_Tepex_WRC1331" in filename else "a scan without a defect"
        }
        data.append({"filename": filename, "label": "defect" if "AMITIE_Tepex_WRC1331" in filename else "healthy"})
        prompts.append(prompt)
    with open(fr'{os.path.basename(source_dir)}_prompt_one_target.json', 'wt') as f:
        f.write(json.dumps({"prompts":prompts}, indent=4))



create_prompt_json(r"/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train")

