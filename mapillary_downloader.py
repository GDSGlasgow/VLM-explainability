# standard packages
import os
import time
import random
import requests
import zoneinfo as zi
from datetime import datetime
from requests.models import Response
# third party packages
import mapillary.interface as mly
import pandas as pd
from tqdm import tqdm
from geopy import geocoders
from sun_position_calculator import SunPositionCalculator, SunPhase

random.seed(2025)


def set_mapillary_token(path:str):
    """Sets the mapillary token in the given path to the os environment

    Args:
        path (str): path to mapillary token. 
    """
    with open(path, 'r') as f:
        token = f.read()
        
    mly.set_access_token(token)
    os.environ['MAPILLARY_TOkEN'] = token

def set_geocoder(user_file:str):
    global GEOCODER

    with open(user_file, 'r') as f:
        username = f.read()
    
    GEOCODER = geocoders.GeoNames(username=username)


def load_sample_coords(filename:str, expected_cols:list[str]=['latitude','longitude','city'])->pd.DataFrame:
    """Retrives the sample coordinates form the provided csv file

    Args:
        filename (str): path to csv file containing "latitude" and "longitude" 
        column describing coordinates to smaple and a "city" column describing 
        the city in whihc the coords are located.
        
        expected_cols (list[str]): a list of the coluns expected in the dataframe.

    Returns:
        pd.DataFrame:detailing the coordinates
    """
    try:
        df = pd.read_csv(filename)
    except:
        print('Unable to access coordinates data.')
        return
    # check columns exist and are formatted properly
    if not all([x for x in df.columns if x in expected_cols]):
        raise ValueError(f'Expected {expected_cols} in dataframe.')
    
    return df
        
        
def get_local_time(timestamp:int, coordinates:tuple[float, float])->datetime:
    """takes an integer timestamp and a set of coordinates (in decimal degrees)
    and returns the local time at that timestamp

    Args:
        timestamp (int): integer timestamp.
        coordinates (tuple[float, float]): in decimal degrees (lat,lon).

    Returns:
        datetime: datetime at the given timestamp and coordinates. 
    """

    tz = zi.ZoneInfo(GEOCODER.reverse_timezone(coordinates).pytz_timezone.zone)
    return datetime.fromtimestamp(timestamp*1e-3, tz=tz)


def get_sunrise_sunset(latitude:float, longitude:float, timestamp:int)->tuple[int, int]:
    """Gets the sunrise and sunset times at the given coordinates and timestamp. 

    Args:
        latitude (float): in decimal degrees.
        longitude (float): in demical degrees.
        timestamp (int): unix timestamp in ms. 
        
    Returns:
        tuple[int,int]: timestamps (in ms) for sunrise and sunset, respectively. 
    """
    calculator = SunPositionCalculator()
    sunrise_time = calculator.time_at_phase(timestamp, SunPhase.sunrise(), latitude, longitude, height=0)
    sunset_time = calculator.time_at_phase(timestamp, SunPhase.sunset(), latitude, longitude, height=0)
    return sunrise_time, sunset_time
    
    
def get_sequence_from_coordinates(coordinates:tuple[float,float])->str:
    """Gets a relevant mapillary sequence from a given string. It wil check that 
    the time stamp corresponds to a locally daytime image. 

    Args:
        coordinates (tuple[float,float]): strcutured as (latitude, longitude)
        in decimal degrees. 

    Returns:
        str: a single sequence_id
    """
    #print(coordinates)
    lat, lon = coordinates[0], coordinates[1]

    data = mly.get_image_close_to(latitude=lat, longitude=lon).to_dict()

    for d in data['features']:
        timestamp = d['properties']['captured_at']
        sunrise, sunset = get_sunrise_sunset(lat, lon, timestamp)
        # check no more than +/- 1 hr from sunrise/sunset
        if sunrise - 60*60*1000 < timestamp < sunset + 60*60*1000:
            return d['properties']['sequence_id'] 
        
            


def get_with_retries(url:str, retries:int=3, delay:int=5, **kwargs):
    """gets the output from an url over a specified number of retries.

    Args:
        url (_type_): url to be searched.
        retries (int, optional): Number of retries to try. Defaults to 3.
        delay (int, optional): Time between retries (seconds). Defaults to 5.

    Returns:
        response from url. 
    """
    for i in range(retries):
        try:
            response = requests.get(url, timeout=60, **kwargs)
            response.raise_for_status()
            return response
        except (requests.exceptions.ProxyError, requests.exceptions.RequestException) as e:
            print(f"Network error ({os.path.basename(url)}): {e}. {delay} are trying again... ({i+1}/{retries})")
            time.sleep(delay)
    print(f"{os.path.basename(url)} adresi {retries} failed. Skipping.")
    return None

def build_url(base_url:str="https://graph.mapillary.com/image_ids", **kwargs)->str:
    """Builds a Mapillary API url on the given base_url using the access token 
    stored in os.environ[MAPILLARY_TOKEN]. Adds kwargs as extra query terms, 
    for example adding argument `sequence_id=123` will add that to the query.
    
    Args:
        base_url (str): The base url to be used. Defaults to standard mapillary. 
        
    Returns:
        str : formatted url for Mapillary API. 
        
    NOTE: This function does not check if the added query terms will break!

    """
    # add access token
    url = base_url + f'?access_token={os.environ["MAPILLARY_TOKEN"]}'
    for k,v in kwargs.items():
        url += f'&{k}={v}'
    return url

def save_image(image_response:Response, save_path:str, image_id:str, chunk_size:int=8192):
    image_filename = os.path.join(save_path, f"{image_id}.jpg")
    # save each image with ID as filename to directory by sequence ID
    with open(image_filename, 'wb') as handler:
        for chunk in image_response.iter_content(chunk_size=chunk_size):
            handler.write(chunk)
            
def build_metadata_row(image_id:str, sequence_id:str, data:dict)->dict:
    lon, lat = data['geometry']['coordinates']
    timestamp = data['captured_at']
    local_time = get_local_time(timestamp, (lat, lon))
    metadata_row = {'image_id':image_id, 
                'sequence_id':sequence_id,
                'lon':lon, 
                'lat':lat,
                'image_url':data['thumb_2048_url'], 
                'timestamp':timestamp,
                'local_time':local_time}
    return metadata_row

def download_sequence_images(sequence_id:int, save_path:str, limit:int=10):
    
    metadata = []
    
    seq_url = build_url(sequence_id=sequence_id, fields='name')
    seq_out = requests.get(seq_url)
    image_ids = [d['id'] for d in seq_out.json()['data']]
    print(f'Dowloading images for sequence {sequence_id}')
    for image_id in tqdm(image_ids[:limit]):
        
        url = build_url(base_url=f'https://graph.mapillary.com/{image_id}', 
                        fields='thumb_2048_url,captured_at,geometry')
        r = get_with_retries(url)
        if not r:
            continue
        data = r.json()
        if 'thumb_2048_url' in data:
            # try to load image
            image_url = data['thumb_2048_url']
            image_response = get_with_retries(image_url, stream=True)
            if not image_response:
                # skip if no response
                continue
            # save each image with ID as filename to directory by sequence ID
            save_image(image_response, save_path, image_id)
            # add metadata
            metadata.append(build_metadata_row(image_id, sequence_id, data))
            
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(save_path, 'metadata.csv'), index=False)
    
            
        
    
def main(coordinates_file:str, token_file:str, user_file:str):
    """Runs full dowload sequence.

    Args:
        coordinates_file (str): csv file path containing coordinates to download.
        token_file (str): path to mapillary token file.
        user_file (str): path to geonames username file.
    """

    set_mapillary_token(token_file)
    set_geocoder(user_file)
    
    query_data = load_sample_coords(coordinates_file)
    
    for i, row in query_data.iterrows():
        coordinates = (row.latitude, row.longitude)
        sequence_id = get_sequence_from_coordinates(coordinates)
        save_path = os.path.join('downloaded_images', row.city, sequence_id)
        # make the directory if it doesn't exist already
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # don't download new data if folder is already populated
        if len(os.listdir(save_path)) == 0:
            download_sequence_images(sequence_id=sequence_id, save_path=save_path, limit=None)
        
if __name__ == '__main__':
    main(coordinates_file='sample_coordinates.csv', 
         token_file='keys/MAPILLARY_TOKEN.txt',
         user_file='keys/GEONAMES_USER.txt')
        