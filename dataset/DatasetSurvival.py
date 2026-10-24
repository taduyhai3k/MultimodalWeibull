import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SurvivalDataset(Dataset):
    def __init__(self, split_key, fold, study_name, omic_format, patient_dict, metadata,omic_data_dict,data_dir, num_classes, label_col, censorship_var, valid_cols, is_training, clinical_data, num_pathces,
                 omic_name, sample, mode):
        super(SurvivalDataset, self).__init__()
        self.split_key = split_key
        self.fold = fold
        self.study_name = study_name
        self.omic_format = omic_format
        self.patient_dict = patient_dict
        self.metadata = metadata
        self.omic_data_dict = omic_data_dict
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_training = is_training
        self.clinical_data = clinical_data
        self.num_patches = num_pathces
        self.omic_name = omic_name
        self.sample = sample
        self.mode = mode
        self._slide_cls_id_prep()

    def get_valid_cols(self):
        return self.valid_cols
    def _slide_cls_id_prep(self):
        self.slide_cls_id = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_id[i] = np.where(self.metadata['label']== i)[0].tolist()
    def _load_wsi_embs_from_path(self, data_dir, slide_ids):
            """
            Load patch embeddings từ các file .pt (Swin Transformer).
            Hỗ trợ Padding và Masking cố định theo num_patches.
            """
            patch_features = []
            
            # 1. Load tất cả các slide .pt của bệnh nhân và gộp lại (Concatenate)
            for slide_id in slide_ids:
                # Loại bỏ đuôi .svs nếu có để khớp với tên file .pt
                clean_id = slide_id.split('.svs')[0]
                wsi_path = os.path.join(data_dir, 'pt_files', f'{clean_id}.pt')
                
                try:
                    wsi_bag = torch.load(wsi_path) # Shape: [N, 768]
                    patch_features.append(wsi_bag)
                except FileNotFoundError:
                    print(f"⚠️ Cảnh báo: Không tìm thấy file {wsi_path}")

            # Nếu không tìm thấy file nào, trả về Tensor trống (tránh crash)
            if len(patch_features) == 0:
                return torch.zeros((self.num_patches, 768)), torch.ones(self.num_patches)

            # Gộp tất cả các slide thành 1 túi (Bag) duy nhất
            patch_features = torch.cat(patch_features, dim=0)
            num_actual_patches = patch_features.shape[0]

            # 2. Xử lý Lấy mẫu (Sampling) hoặc Padding
            if self.sample:
                max_p = self.num_patches
                
                if num_actual_patches >= max_p:
                    # Trường hợp đủ hoặc thừa: Lấy mẫu ngẫu nhiên không lặp lại
                    idx = np.random.choice(num_actual_patches, max_p, replace=False)
                    idx = np.sort(idx) # Sắp xếp lại index để giữ tính ổn định không gian (tùy chọn)
                    patch_features = patch_features[idx]
                    mask = torch.zeros(max_p) # Tất cả đều là data thật (Mask = 0)
                else:
                    # Trường hợp thiếu: Giữ nguyên data thật + Padding thêm số 0
                    num_pad = max_p - num_actual_patches
                    pad_zeros = torch.zeros((num_pad, patch_features.shape[1]))
                    patch_features = torch.cat([patch_features, pad_zeros], dim=0)
                    
                    # Mask: 0 cho data thật, 1 cho data padding
                    mask = torch.cat([torch.zeros(num_actual_patches), torch.ones(num_pad)])
                if self.mode == 'shuffle':
                    # Nếu đang ở chế độ shuffle, trộn cả patch_features và mask cùng nhau
                    perm = torch.randperm(max_p)
                    patch_features = patch_features[perm]
                    mask = mask[perm]
            
            else:
                # Nếu KHÔNG dùng sample: Trả về toàn bộ data hiện có
                # Mask lúc này chỉ đơn giản là một mảng toàn số 0 (vì toàn bộ là data thật)
                mask = torch.zeros(num_actual_patches)

            return patch_features, mask
    def get_clinical_data(self, case_id):
        """
        Truy xuất thông tin lâm sàng (Stage, Grade, Subtype) dựa trên ID bệnh nhân.

        Args:
            case_id (str): Mã định danh duy nhất của bệnh nhân.

        Returns:
            tuple: Bộ 3 giá trị (stage, grade, subtype). 
                   Trả về ("N/A", "N/A", "N/A") nếu ID không tồn tại hoặc dữ liệu trống.
        """
        # 1. Kiểm tra nhanh xem ID có trong bảng lâm sàng không
        if case_id not in self.clinical_data["case_id"].values:
            return ("N/A", "N/A", "N/A")

        # 2. Lấy hàng dữ liệu của bệnh nhân
        # Dùng .loc một lần duy nhất để tiết kiệm thời gian truy xuất
        patient_info = self.clinical_data.loc[self.clinical_data["case_id"] == case_id].iloc[0]
        # 3. Trích xuất giá trị và xử lý NaN (dữ liệu trống trong file CSV)
        # Hàm xử lý nhanh: Nếu giá trị là NaN hoặc không tồn tại thì gán "N/A"
        def clean_val(val):
            return "N/A" if pd.isna(val) or val == "" else val

        # Lấy các cột tương ứng (Hải nhớ kiểm tra tên cột trong file CSV có đúng là stage, grade, subtype không nhé)
        stage = clean_val(patient_info.get("stage", "N/A"))
        grade = clean_val(patient_info.get("grade", "N/A"))
        subtype = clean_val(patient_info.get("subtype", "N/A"))

        return (stage, grade, subtype)
    def get_data_to_return(self, idx):
        """
        Gom tất cả metadata và thông tin slide để trả về cho 1 case_id cụ thể.
        """
        case_id = self.metadata['case_id'].iloc[idx]
        # Chuyển label, time, censorship sang Tensor
        label = torch.tensor(self.metadata['label'].iloc[idx], dtype=torch.long)
        event_time = torch.tensor([self.metadata[self.label_col].iloc[idx]], dtype=torch.float32)
        c = torch.tensor([self.metadata[self.censorship_var].iloc[idx]], dtype=torch.float32)
        
        # Tra cứu slide_ids từ patient_dict (trả về list các file .svs/.pt)
        slide_ids = self.patient_dict.get(case_id, [])
        
        # Lấy bộ 3 lâm sàng (stage, grade, subtype)
        clinical_data = self.get_clinical_data(case_id)

        return label, event_time, c, slide_ids, clinical_data, case_id

    def __getitem__(self, idx):
        """
        Hàm chính để DataLoader lấy dữ liệu.
        """
        # 1. Lấy dữ liệu cơ bản
        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)

        # 2. Xử lý dữ liệu Omics dựa trên omic_format
        # Chú ý: Dùng .iloc[idx] vì các bảng đã được đồng bộ hóa từ trước
        
        if self.omic_format == "gene":
            # Lấy toàn bộ vector gene của bệnh nhân
            # Loại bỏ cột 'temp_index' để chỉ còn thuần túy dữ liệu số
            df_small = self.omic_data_dict.iloc[idx].drop(labels="temp_index", errors='ignore')
            # Sắp xếp tên gene theo bảng chữ cái để đảm bảo thứ tự luôn cố định
            df_small = df_small.sort_index()
            omics_tensor = torch.tensor(df_small.values.astype(float), dtype=torch.float32)

            # Trả về dummy ảnh (zeros) vì format gene thường không dùng WSI hoặc dùng sau
            return (torch.zeros((1, 1)), omics_tensor, label, event_time, c, clinical_data)

        elif self.omic_format == "groups":
            # Load WSI đặc trưng và mask
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            
            omics_groups = []
            for i in range(len(self.omic_name)):
                group_data = self.omic_data_dict[self.omic_name[i]].iloc[idx]
                omics_groups.append(torch.tensor(group_data.values.astype(float), dtype=torch.float32))
            
            return (patch_features, *omics_groups, label, event_time, c, clinical_data, mask)

        elif self.omic_format == "pathways":
            # Load WSI đặc trưng và mask
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)

            # Lấy danh sách các pathway
            omic_list = []
            for i in range(len(self.omic_name)):
                # Lấy dữ liệu của pathway thứ i cho bệnh nhân hiện tại
                pathway_data = self.omic_data_dict[self.omic_name[i]].iloc[idx]
                omic_list.append(torch.tensor(pathway_data.values.astype(float), dtype=torch.float32))

            return (patch_features, omic_list, label, event_time, c, clinical_data, mask)

        else:
            raise NotImplementedError(f'Omics Type [{self.omic_format}] chưa được hỗ trợ.')

    def __len__(self):
        return len(self.metadata)
    def get_label(self, idx):
        return self.metadata["label"].iloc[idx]



