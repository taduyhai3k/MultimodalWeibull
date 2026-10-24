import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from . import DatasetSurvival

class SurvivalDatasetFactory:
    def __init__(self, label_file: str, omics_file: str, clinical_file: str, signature_file: str,wsi_dir:str,study_name: str,omic_format:str,seed: int = 42, n_bins: int = 4, label_col: str = "survival_months_dss", print_info: bool = True, eps: float = 1e-8, num_patches: int = 4096, 
                  mode: str = "train") -> None:
        r"""SurvivalDatasetFactory is a class that creates a survival dataset for training and evaluation.
        Args:
            label_file (str): The path to the label file that contains survival information.
            omics_file (str): The path to the omics file that contains the feature data generation of patients.
            clinical_file (str): The path to the clinical file that contains additional patient information.
            signature_file (str): The path to the signature file.
            wsi_dir (str): The directory containing the whole slide images.
            study_name (str): The name of the study.
            omic_format (str): The format of the omics data (e.g., "pathway", "gene", etc.).
            seed (int): The random seed for reproducibility.
            n_bins (int): The number of bins for discretizing the survival time.
            label_col (str): The column name for the survival label.
            print_info (bool): Whether to print information about the dataset.
            eps (float): A small value to avoid division by zero.
            num_patches (int): The number of patches to sample from each image.
            mode (str): The mode of the dataset ("train" or "test").
        """
        self.label_file = label_file
        self.omics_file = omics_file
        self.clinical_file = clinical_file
        self.signature_file = signature_file
        self.wsi_dir = wsi_dir
        self.study_name = study_name
        self.omic_format = omic_format
        self.seed = seed
        self.n_bins = n_bins
        self.label_col = label_col
        self.print_info = print_info
        self.eps = eps
        self.num_patches = num_patches
        self.mode = mode

        if self.label_col == "survival_months":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"
        self._read_omics_data()
        self._read_label_data()
        self._read_clinical_data()
        self._read_signature_data()
        self._discretize_labels()
        self._init_patients_data()
        self._init_label_discrete_dict()
        self._init_slide_id_dict()
        self._init_label_index_map()
    
    def _read_signature_data(self):
        r"""Read the signature data and group by intersection of this and omics_data"""
        self.omic_names = []
        df_omics = pd.read_csv(self.signature_file)
        for col in df_omics.columns:
            omic = df_omics[col].dropna().unique()
            omic = sorted(pd.Series(list(set(omic) & set(self.omics_data.columns))))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def _read_clinical_data(self):
        r"""Loads the clinical data from the specified file and processes it for use in the dataset.
        """
        self.clinical_data = pd.read_csv(self.clinical_file, index_col=0)

    def _read_omics_data(self):
        r"""Reads the omics data from the specified file and processes it for use in the dataset.
        """
        # Read the omics data from the file
        omics_data = pd.read_csv(self.omics_file, index_col=0, engine="python")
        self.omics_data = omics_data
    def _read_label_data(self):
        r"""Reads the label data from the specified file and processes it for use in the dataset.
        """
        # Read the label data from the file
        label_data = pd.read_csv(self.label_file, low_memory=False)

        if "IDC" in label_data['oncotree_code']:  # must be BRCA (and if so, use only IDCs)
            label_data = label_data[label_data['oncotree_code'] == 'IDC']
        label_data = label_data.drop_duplicates(["case_id"])
        self.label_data = label_data
    def _get_uncensored_data_df(self):
        r"""Gets the uncensored data from the label data and returns it as a DataFrame.
        """
        # Get the uncensored data
        uncensored_data_df = self.label_data[self.label_data[self.censorship_var] < 1]
        return uncensored_data_df
    def _discretize_labels(self):
        r"""Discretizes the survival labels into bins based on the specified number of bins and the label column.
        """
        # Discretize the survival labels into bins
        uncensored_data_df = self._get_uncensored_data_df() 
        dis_label, q_bin = pd.qcut(uncensored_data_df[self.label_col], q=self.n_bins, labels=False, retbins=True, duplicates="drop")
        q_bin[-1] = self.label_data[self.label_col].max() + self.eps
        q_bin[0] = self.label_data[self.label_col].min() - self.eps
        if self.print_info:
            print(f"Quantile bins: {q_bin}")
        new_label,new_bin = pd.cut(self.label_data[self.label_col], bins=q_bin, labels=False, retbins=True, include_lowest=True, right=False)
        self.label_data.insert(2, "label", new_label.values.astype(int))
    def _init_patients_data(self):
        r"""Initializes the patient data with columns case_id and label discretized into bins.
        """
        patients_data = self.label_data[~self.label_data["case_id"].duplicated()]
        self.patients_data = {"case_id": patients_data["case_id"].values, "label": patients_data["label"].values}
    def _init_label_discrete_dict(self):
        r"""Initializes a dicttionary that maps tuple(bin, cencorship) to index and add the column "label_discrete" to the label data and convert label column to index.
        """      
        label_discrete_dict = {}
        index = 0
        for i in range(self.n_bins):
            for c in [0, 1]:
                label_discrete_dict[(i, c)] = index
                index += 1
        print("Các key hiện có trong dict:", label_discrete_dict.keys())
        print("Giá trị label lớn nhất trong data:", self.label_data["label"].max())
        print("Giá trị censorship duy nhất:", self.label_data[self.censorship_var].unique())
        print(label_discrete_dict)
        self.label_data.insert(3, "label_discrete", self.label_data.apply(lambda x: label_discrete_dict[(x["label"], x[self.censorship_var])], axis=1))
        #self.label_data["label"] = self.label_data.apply(lambda x: label_discrete_dict[(x["label"], x[self.censorship_var])], axis=1)
        self.label_discrete_dict = label_discrete_dict
    def _init_slide_id_dict(self):
        r"""Initializes a dictionary that maps case_id to slide_id for each patient.
        """
        slide_id_dict = {}
        for _, row in self.label_data.iterrows():
            case_id = row["case_id"]
            slide_id = row["slide_id"]
            if case_id not in slide_id_dict:
                slide_id_dict[case_id] = []
            slide_id_dict[case_id].append(slide_id)
        self.slide_id_dict = slide_id_dict     
    def _init_label_index_map(self):
        r"""Initializes a list that contains lists of indices for each label in the dataset.
        """
        label_index_map = [[] for _ in range(self.n_bins * 2)]
        for index, row in self.label_data.iterrows():
            label = row["label"]
            label_index_map[label].append(index)
        self.label_index_map = label_index_map
        slide_index_map = [[] for _ in range(self.n_bins * 2)]
        for index, row in self.label_data.iterrows():       
            slide_id = row["slide_id"]
            label = row["label"]
            slide_index_map[label].append(slide_id)
        self.slide_index_map = slide_index_map
    def _get_scaler(self, data):
        r"""Gets a MinMaxScaler fitted to the provided data.
        Args:
            data (pd.DataFrame): The data to fit the scaler on.
        Returns:
            MinMaxScaler: A fitted MinMaxScaler object.
        """
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(data)
        return scaler
    def _apply_scaler(self, data, scaler):
        r"""Applies the provided scaler to the data, while keeping zero values unchanged."""
        zero_mask = (data == 0)
        scaled_data = scaler.transform(data)
        scaled_data[zero_mask] = 0
        return scaled_data

    def _get_split_from_df(self, split_path:str, split_key="train", fold=None, scaler=None, valid_cols=None):
            """
            Lấy dữ liệu split và đồng bộ hóa tuyệt đối giữa Omics, Clinical và Metadata.
            """
            if not scaler:
                scaler = None
            df_split = pd.read_csv(split_path)
            # 1. Lấy danh sách ID từ file split (loại bỏ NaN)
            split_id_series = df_split[split_key].dropna()
            split_set = set(split_id_series.tolist())

            # 2. Tìm "Giao điểm" của 4 nguồn dữ liệu để tìm ra các ID hợp lệ nhất
            # - Có trong split hiện tại
            # - Có trong dữ liệu Omics
            # - Có trong dữ liệu Lâm sàng (Clinical)
            # - Có trong dữ liệu nhãn (Label/Metadata)
            omics_ids = set(self.omics_data.index.tolist())
            clinical_ids = set(self.clinical_data["case_id"].tolist())
            label_ids = set(self.label_data["case_id"].tolist())

            # Phép giao tập hợp (&) cực nhanh
            common_ids = list(split_set & omics_ids & clinical_ids & label_ids)
            common_ids.sort() # Sắp xếp để đảm bảo tính tái lập (Reproducibility)

            if len(common_ids) == 0:
                print(f"⚠️ Cảnh báo: Không tìm thấy ID chung nào cho split {split_key}!")
                return None, None

            # --- BẮT ĐẦU TRÍCH XUẤT VÀ ĐỒNG BỘ ---

            # 3. Lọc Metadata (df_data)
            df_data = self.label_data[self.label_data["case_id"].isin(common_ids)].copy()
            df_data.set_index("case_id", inplace=True)
            df_data = df_data.loc[common_ids] # Ép đúng thứ tự common_ids
            df_data.reset_index(inplace=True) # Đưa case_id trở lại thành cột
            df_data.reset_index(drop=True, inplace=True)

            # 4. Lọc Omics (df_omics)
            # .loc[common_ids] đảm bảo lấy đúng ID và ĐÚNG THỨ TỰ
            df_omics = self.omics_data.loc[common_ids, :].copy()
            
            df_omics = df_omics[~df_omics.index.duplicated(keep="first")]
            case_ids_series = df_omics.index.copy()
            # Tạo temp_index để lưu ID gốc trước khi reset_index

            # 5. Lọc Clinical (df_clinical)
            df_clinical = self.clinical_data.copy()
            df_clinical.set_index("case_id", inplace=True) # Đảm bảo case_id là index để lọc nhanh
            df_clinical = df_clinical.loc[common_ids] # Ép đúng thứ tự common_ids
            df_clinical.reset_index(inplace=True) # Đưa case_id trở lại thành cột
            df_clinical = df_clinical[~df_clinical.index.duplicated(keep="first")]
            # Sửa lỗi: fillna không gán lại khi dùng inplace=True
            df_clinical.fillna("N/A", inplace=True)
            # Đảm bảo index lâm sàng là case_id để SurvivalDataset dễ truy xuất
            df_clinical.index.name = "case_id" 

            # case_id = df_omics["temp_index"]
            # df_omics.drop(columns=["temp_index"])
            num_patients, num_features = df_omics.shape
            original_columns = df_omics.columns
            flat_omics = df_omics.values.flatten().reshape(-1,1)
            print(flat_omics)
            if split_key == "train":
                scaler = self._get_scaler(flat_omics)
            else:
                if scaler is None:
                    raise ValueError("Scaler must be provided for non-training splits.")
            norm_omics = self._apply_scaler(flat_omics, scaler)
            df_omics = pd.DataFrame(norm_omics.reshape(num_patients, num_features), columns=original_columns)
            df_omics["temp_index"] = case_ids_series.values
            sample = (split_key == "train")
            
            dataset = DatasetSurvival.SurvivalDataset(split_key=split_key, fold= fold, study_name= self.study_name, omic_format= self.omic_format, patient_dict=self.slide_id_dict,
                                                      metadata= df_data, omic_data_dict= df_omics, data_dir= self.wsi_dir, num_classes= len(self.label_discrete_dict), 
                                                      label_col= self.label_col, censorship_var= self.censorship_var, valid_cols= valid_cols, is_training= sample, clinical_data= df_clinical,
                                                      num_pathces=self.num_patches, omic_name= self.omic_names, sample= sample, mode= self.mode)
            if split_key == "train":
                return dataset, scaler
            else:
                return dataset, None
        


        