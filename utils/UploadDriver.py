import subprocess
import os

def upload_to_drive(local_path, drive_folder="WSI_Project/features"):
    """
    local_path: Đường dẫn file hoặc thư mục trên máy Vast.ai
    drive_folder: Thư mục đích trên Google Drive
    """
    # Lệnh rclone: copy [nguồn] [tên_remote]:[đích]
    # --drive-chunk-size 64M giúp upload file lớn nhanh hơn
    cmd = [
        "rclone", "copy", 
        local_path, 
        f"mydrive:{drive_folder}",
        "--drive-chunk-size", "64M",
        "--threads", "4"
    ]
    
    try:
        print(f"--- Đang tải lên Drive: {local_path} ---")
        subprocess.run(cmd, check=True)
        print("--- Tải lên thành công! ---")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi Rclone: {e}")