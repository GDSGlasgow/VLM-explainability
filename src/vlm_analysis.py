import torch
from pathlib import Path

from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torchvision.transforms as T
import numpy as np


from vlm_utils import load_model, load_image
import json_repair


class ImageAnalyzer:
    
    def __init__(self, model, tokenizer, model_config):
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        
    def analyze_image(self, mask_path:str|Path, question:str)->str:
        """Loads an image from `mask_path` and passes it to the VLM with the 
        prompt in `question`.
        
        Args:
            mask_path (str|Path) : path to the image to be analysed
            question (str) : Prompt for the model. 
            
        Returns:
            str : the response form the model.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pixel_values = load_image(mask_path, max_num=12).to(torch.bfloat16).to(device)
        response = self.model.chat(self.tokenizer, pixel_values, question, self.model_config)
        return response
    
    def analyze_all_masks(self, image_dir: Path, question: str) -> tuple[dict, dict]:
        """Passes all the images in a provided directory and parses them using 
        the VLM.
        
        Args:
            image_dir (str|Path) : path to the images to be analysed.
            question (str) : Prompt for the model. 
        Returns:
            tuple[dict,dict] : (scores, repsonses) from the model. 
        """
        mask_paths = list(image_dir.glob('*.jpg'))
        vlm_outputs = {}
        vlm_scores = {}

        for mask_path in tqdm(mask_paths, leave=False):
            label = mask_path.stem
            response = self.analyze_image(mask_path, question)
            vlm_outputs[label] = response
            vlm_scores[label] = self.extract_scores(response)

        return vlm_scores, vlm_outputs
    
    def extract_scores(self, vlm_output:str)-> int|str:
        """Extracts the score from the JSON within the VLM output
        
        Args:
            vlm_output (str) : the output provided byt the model.
        Returns:
            int : a score extracted from a JSON structure wihtin the ouptut
            str : "No score provided" if a json could nto be extracted.
        """
        if '```' in vlm_output:
            json_part = vlm_output.split('```')[1].split('```')[0]
            json_out = json_repair.loads(json_part)
        else:
            json_out = json_repair.loads(vlm_output)
        try:
            return json_out[0]['estimated_score']
        except KeyError:
            return json_out[0].values[0]
        except IndexError:
            return 'No Score Provided'

class ResultsManager:
    def __init__(self, seq_path: Path):
        self.seq_path = seq_path
        self.metadata = pd.read_csv(Path(seq_path, 'metadata.csv'))
        self.metadata['image_id'] = self.metadata['image_id'].astype(str)
        self.results = []

    def add_image_results(self, image_id: str, scores: dict, responses: dict):
        """Adds the scores and responses for all mask combinations in a single
        image to the full ist of results
        
        Args:
            image_id (str) : The unique id for the image to be analysed. 
            scores (dict[str,int]) : Scores for each combination (labels).
            responses (dict[str,str]) : Responses for each combination.  
        Updates:
            self.results : Appends a pd.DataFrame with the image results. 
        """
        image_metadata = self.metadata[self.metadata.image_id == image_id].copy()
        if image_metadata.empty:
            raise ValueError(f"No metadata found for image_id {image_id}")

        image_metadata['score'] = image_metadata['label'].apply(lambda label: scores.get(label, np.nan))
        image_metadata['response'] = image_metadata['label'].apply(lambda label: responses.get(label, np.nan))

        image_metadata = self.add_unmasked_row(image_metadata, image_id, scores, responses)
        self.results.append(image_metadata)

    def add_unmasked_row(self, df:pd.DataFrame, image_id:str, scores:dict, responses:dict) -> pd.DataFrame:
        """
        Appends a new row representing the unmasked image, using available score/response.
        
        Args:
            df (pd.DataFrame) : Data to have new row added. 
            image_id (str) : Unique ID for current image.
            scores (dict[str, int]) : score for unmasked image.
            repsonses (dict[str,str]) : response for unmasked image. 
        Returns:
            pd.DataFrame : With row for unmasked data added. 
        """
        base_row = {col: 0 for col in df.columns}
        base_row.update({
            'label': 'unmasked',
            'score': scores.get('unmasked', np.nan),
            'response': responses.get('unmasked', np.nan),
            'image_id': image_id
        })
        return pd.concat([df, pd.DataFrame([base_row])], ignore_index=True)

    def save_results(self, out_path: Path):
        results_df = pd.concat(self.results, ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
    

def main(seq_id, prompt_path):
    
    seq_path = Path('images', 'masks', seq_id)
    
    #metadata = pd.read_csv(Path(seq_path, 'metadata.csv'))
    #metadata['image_id'] = metadata['image_id'].astype(str) 
    
    with open(prompt_path, 'r') as f:
        question = f.read()

    model, tokenizer = load_model("OpenGVLab/InternVL3-38B", quant=True)
    config = dict(max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    analyzer = ImageAnalyzer(model, tokenizer, config)
    results_manager = ResultsManager(seq_path)

    image_paths = [p for p in list(seq_path.glob('**/')) if p != seq_path]
    for path in tqdm(image_paths):
        scores, responses = analyzer.analyze_all_masks(path, question)
        results_manager.add_image_results(path.stem, scores, responses)
        results_manager.save_results(Path(f'results/{seq_id}.csv'))
    
    
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', '--seq_id', 
                        dest='seq_id', 
                        help='Sequence to analysed.')
    
    parser.add_argument('-p', '-prompt_path',
                        dest='prompt_path',
                        help='Path to prompt.')
    
    args = parser.parse_args()
    
    main(args.seq_id, args.prompt_path)