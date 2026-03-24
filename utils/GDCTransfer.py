
import os
import requests
import pandas as pd
import tqdm

class GDC_Downloader:
    """Class tự động tải ảnh từ Manifest"""
    @staticmethod
    def download_from_manifest(manifest_path, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        df = pd.read_csv(manifest_path, sep='\t')
        
        data_endpoint = "https://api.gdc.cancer.gov/data/"
        downloaded_paths = []

        for _, row in df.iterrows():
            uuid, name = row['id'], row['filename']
            target_path = os.path.join(save_dir, name)
            
            if not os.path.exists(target_path):
                print(f"Downloading {name}...")
                response = requests.get(data_endpoint + uuid, stream=True)
                if response.status_code == 200:
                    with open(target_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                else:
                    print(f"Failed to download {name}")
                    continue
            
            downloaded_paths.append(target_path)
        return downloaded_paths
    @staticmethod
    def download_single_file(uuid, filename, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        data_endpoint = "https://api.gdc.cancer.gov/data/"
        response = requests.get(data_endpoint + uuid, stream=True)
        if response.status_code == 200:
            target_path = os.path.join(save_dir, filename)
            with open(target_path, 'wb') as f:
                iterable_response = tqdm.tqdm(response.iter_content(chunk_size=1024*1024), desc=f"Downloading {uuid}")
                for chunk in iterable_response:
                    f.write(chunk)
            return target_path
        else:
            print(f"Failed to download file with UUID: {uuid}")
            return None