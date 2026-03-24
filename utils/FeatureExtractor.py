import torch
import timm
from torchvision import transforms


class FeatureExtractor:
    """Class dùng Swin Transformer để trích xuất đặc trưng"""
    def __init__(self, model_name='swin_tiny_patch4_window7_224'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        
        # Cấu hình chuẩn hóa từ model
        config = self.model.default_cfg
        self.transform = transforms.Compose([
            transforms.Resize(config['input_size'][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std'])
        ])

    def extract(self, img_pil):
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(img_tensor)
        return feature.cpu().float() # Trả về dạng float32 để lưu
    def extract_batch(self, imgs_pil_list):
        """Trích xuất đặc trưng cho một danh sách các ảnh PIL"""
        # Chuyển list PIL Images thành một Tensor Batch duy nhất [N, 3, 224, 224]
        batch_tensor = torch.stack([self.transform(img) for img in imgs_pil_list]).to(self.device)
        
        with torch.no_grad():
            # Chạy inference trên cả batch
            features = self.model(batch_tensor)
        
        return features.cpu().float() # Trả về tensor [N, Feature_Dim]