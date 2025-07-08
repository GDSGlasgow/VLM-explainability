# standard library imports
import requests
import torch
import argparse
from PIL import Image
from pathlib import Path
import os
import json
from json import JSONDecodeError
from typing import Optional
from copy import copy
import itertools

# third party imports
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
from pydantic import BaseModel
import pandas as pd
from tqdm import tqdm


class MaskerConfig(BaseModel):
    vegetation: list[str]
    building: list[str]
    person: list[str]
    traffic: list[str]
    street: list[str]
    sky: Optional[list[str]] = None
    max_combination: int
    model_name: str
    label_data_path: str


class Masker:
    
    def __init__(self, config_path:str, mask_categories):
        
        self.config = self.load_config(config_path)
        self.model, self.processor = self.load_model(self.config.model_name)
        self.mask_categories = mask_categories
        self.labels_df = pd.read_csv(self.config.label_data_path)
        
    def load_config(self, config_path:str|Path):
        """Loads the config JSON from the given path. Note - this does nto"""
        with open(config_path, 'r') as f:
            try:
                config_dict = json.load(f)
            except FileNotFoundError as e:
                raise(e, "Please ensure the config path is correct.")
            except JSONDecodeError as e:
                raise(e, "Please ensure the config JSON is formatted correctly")
            
        return MaskerConfig(**config_dict)
    
    def labels_to_indices(self, labels:list[str])->list[int]:
        
        category_df = self.labels_df[self.labels_df.classes.isin(labels)]
        
        return category_df.idx.values 
    
    def load_model(self, model_name:str):
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        except Exception as e:
            raise(e, "Please ensure model name in config is correct.")
        return model, processor
    
    def segment_image(self, image:Image)->torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    def build_mask(self, semantic_map:torch.Tensor, category:str)->np.array:
        
        # get the CityScape categories for the mask
        cs_labels = self.config.model_dump()[category]
        # and get the indices for those categories
        indices = self.labels_to_indices(cs_labels)
        masks, mask_indices = self.processor.convert_segmentation_map_to_binary_masks(semantic_map)
        # initialise output_arr
        out_arr = np.zeros(shape=masks[0].shape)
        for mask, idx in zip(masks, mask_indices):
            if idx in indices:
                out_arr += mask
        return abs(out_arr-1)
    
    def build_table_row(self, mask_categories:list[str], combo:list[str], masks:dict[str, np.array]):
        table_row = {}
        # add row information for output table
        for c in mask_categories:
            if c in combo:
                # presence of mask
                table_row[c] = 1
                # proportion of mask
                table_row[c+'_p'] = 1 - (np.sum(masks[c])/np.size(masks[c]))
            else:
                table_row[c] = 0
                table_row[c + '_p'] = 0

        # add label information  
        label = '_'.join(combo)
        table_row['label'] = label
        return table_row
    
    def generate_masks(self, image:Image):
        """ Generates a combination of up to 3 of each of the categories of mask"""
        semantic_map = self.segment_image(image)
        print(semantic_map.shape)
        generated_masks = {}
        combo_table = []
        
        for i in range(1, self.config.max_combination + 1):
            # loop over each combination up to max_combinations
            for combo in itertools.combinations(self.mask_categories, i):
                # build a label
                label = '_'.join(combo)
                # get the masks and add to the otuput
                masks = {c:self.build_mask(semantic_map, c) for c in combo}
                mask = np.prod(np.stack(list(masks.values())), axis=0)
                generated_masks.update({label:mask})
                # update table for this mask
                table_row = self.build_table_row(self.mask_categories, combo, masks)
                combo_table.append(table_row)

        combo_df = pd.DataFrame(combo_table)
        return generated_masks, combo_df


def main(seq_path:str, config_path:str|Path):
    """Uses the `Masker` class to produce masks for all images in the `seq_id` 
    folder. Images are saved in `images/masks/seq_id
    
    Args
        seq_path (str) : the path to the images in the sequence. 
        config_path (str|Path) : the path to the config file detailing the 
        constittuents of each masking category.
    """
    # convert sequence path to Path
    if isinstance(seq_path, str):
        seq_path = Path(seq_path)
    # get just the sequence_id
    seq_id = seq_path.parts[-1]
    save_path = Path('images', 'masks', seq_id)
    
    # set up masker
    masker = Masker(config_path, mask_categories = ['building', 'person', 'vegetation', 'street', 'traffic', 'sky'])
    
    metadata = []
    jpg_files = list(seq_path.glob("*.jpg"))
    for f in tqdm(jpg_files):
        # load image
        image= Image.open(f)
        # set the save path (and create directories if necessary)
        mask_path = Path(save_path, f.stem)
        Path(mask_path).mkdir(parents=True, exist_ok=True)
        # create masks
        masks, combo_df = masker.generate_masks(image)
        # jointogether masks and apply to image
        for label, mask in masks.items():
            masked_array = image * np.stack([mask]*3, axis=2)
            # save image as jpg
            masked_image = Image.fromarray(np.uint8(masked_array))
            masked_image.save(Path(mask_path, f'{label}.jpg'))
        
        image.save(Path(mask_path, 'unmasked.jpg'))
            
        combo_df['image_id'] = str(f.stem)
        metadata.append(combo_df)
    
    metadata_df = pd.concat(metadata, ignore_index=True)
    metadata_df.to_csv(Path(save_path, 'metadata.csv'))
            
        
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', 
                        '--seq_path', 
                        dest='seq_path',
                        help='ID for sequence of images to be masked')
    
    parser.add_argument('-c', 
                        '--config', 
                        dest='config',
                        help='Path to JSON file showing class constituents.')
    
    args = parser.parse_args()
    
    main(args.seq_path, args.config)