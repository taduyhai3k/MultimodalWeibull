import FeatureExtractor
import GDCTransfer
import WSI
import ContourCheck
import UploadDriver
import pandas as pd
import os

def get_case_id(filename):
    """
    Trích xuất Case ID (TCGA-XX-XXXX) từ tên file SVS của TCGA.
    Ví dụ: TCGA-3C-AALI-01Z-00-DX1... -> TCGA-3C-AALI
    """
    # Lấy tên file nếu đầu vào là đường dẫn (path/to/file.svs)
    basename = os.path.basename(filename)
    
    # Cách 1: Tách theo dấu gạch ngang và lấy 3 phần đầu
    # TCGA (0), 3C (1), AALI (2)
    parts = basename.split('-')
    if len(parts) >= 3:
        case_id = "-".join(parts[:3])
        return case_id
    
    # Cách 2: Nếu cấu trúc file lạ, lấy đúng 12 ký tự đầu (chuẩn TCGA)
    return basename[:12]
def process_one_svs(id:str, filename:str, extractor:FeatureExtractor.FeatureExtractor, patch_size:int, final_size:int, batch_size:int, driver_folder:str):
    svs_path = GDCTransfer.GDC_Downloader.download_single_file(id, filename, "temp_download")
    wsi = WSI.EnhancedWSI(svs_path, patch_size, final_size, level=0)
    wsi.process_and_extract(extractor=extractor, batch_size=batch_size, save_feature_dir="temp_features")
    UploadDriver.upload_to_drive("temp_features", drive_folder=os.path.join(driver_folder, get_case_id(filename)))
    # Xóa file tạm sau khi upload
    os.remove(svs_path)
    for f in os.listdir("temp_features"):
        os.remove(os.path.join("temp_features", f))
def process_manifest(manifest_path:str, extractor:FeatureExtractor.FeatureExtractor, patch_size:int, final_size:int, batch_size:int, driver_folder:str):
    df = pd.read_csv(manifest_path, sep='\t')
    for _, row in df.iterrows():
        id, filename = row['id'], row['filename']
        process_one_svs(id, filename, extractor, patch_size, final_size, batch_size, driver_folder)
def process_manifestes(list_of_manifest_paths:list, model_name:str, patch_size:int, final_size:int, batch_size:int, driver_folder:str):
    extractor = FeatureExtractor.FeatureExtractor(model_name=model_name)
    for manifest_path in list_of_manifest_paths:
        print(f"--- Processing manifest: {manifest_path} ---")
        process_manifest(manifest_path, extractor, patch_size, final_size, batch_size, driver_folder + "/" + os.path.basename(manifest_path).split('.')[0])
    