# standard library imports
import requests
import torch
import argparse
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
import os
import json
from json import JSONDecodeError
from typing import Optional
from copy import copy
import itertools


DEFAULT_BLUR_RADIUS = 18
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
        label_data_path = Path(self.config.label_data_path)
        if not label_data_path.is_absolute():
            label_data_path = PROJECT_ROOT / label_data_path
        self.labels_df = pd.read_csv(label_data_path)
        
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
        """Generate masks for every configured combination of categories."""
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


def blur_radius(value: str) -> int:
    """Parse and validate a Gaussian blur radius from the command line."""
    try:
        radius = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("blur radius must be an integer") from exc
    if not 0 <= radius <= 100:
        raise argparse.ArgumentTypeError("blur radius must be between 0 and 100")
    return radius


def build_mask_fill(
    image: Image.Image,
    greyscale: bool = True,
    radius: int = DEFAULT_BLUR_RADIUS,
) -> Image.Image:
    """Build the image used to replace pixels selected by a semantic mask."""
    if not 0 <= radius <= 100:
        raise ValueError("blur radius must be between 0 and 100")

    image = image.convert("RGB")
    mask_fill = ImageOps.grayscale(image).convert("RGB") if greyscale else image
    return mask_fill.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_mask(
    image: Image.Image,
    mask: np.ndarray,
    mask_fill: Image.Image,
) -> Image.Image:
    """Replace masked pixels with pixels from ``mask_fill``."""
    image = image.convert("RGB")
    keep_mask = Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255), mode="L")
    return Image.composite(image, mask_fill, keep_mask)


def main(
    seq_path: str,
    config_path: str | Path,
    greyscale: bool = True,
    radius: int = DEFAULT_BLUR_RADIUS,
):
    """Uses the `Masker` class to produce masks for all images in the `seq_id` 
    folder. Images are saved in `images/masks/seq_id
    
    Args
        seq_path (str) : the path to the images in the sequence. 
        config_path (str|Path) : the path to the config file detailing the 
        constittuents of each masking category.
        greyscale (bool) : whether masked areas should be converted to greyscale.
        radius (int) : Gaussian blur radius, from 0 to 100 pixels.
    """
    # convert sequence path to Path
    if isinstance(seq_path, str):
        seq_path = Path(seq_path)
    # get just the sequence_id
    seq_id = seq_path.parts[-1]
    save_path = PROJECT_ROOT / 'images' / 'masks' / seq_id
    
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
        mask_fill = build_mask_fill(image, greyscale=greyscale, radius=radius)
        # Apply each semantic mask to the image.
        for label, mask in masks.items():
            masked_image = apply_mask(image, mask, mask_fill)
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

    greyscale_group = parser.add_mutually_exclusive_group()
    greyscale_group.add_argument(
        '--greyscale',
        dest='greyscale',
        action='store_true',
        help='Convert masked areas to greyscale (default).',
    )
    greyscale_group.add_argument(
        '--no-greyscale',
        dest='greyscale',
        action='store_false',
        help='Keep colour in masked areas.',
    )
    parser.set_defaults(greyscale=True)

    parser.add_argument(
        '--blur-radius',
        type=blur_radius,
        default=DEFAULT_BLUR_RADIUS,
        help=f'Gaussian blur radius from 0 to 100 (default: {DEFAULT_BLUR_RADIUS}).',
    )
    
    args = parser.parse_args()
    
    main(
        args.seq_path,
        args.config,
        greyscale=args.greyscale,
        radius=args.blur_radius,
    )
