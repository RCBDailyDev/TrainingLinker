"""
FILE_NAME: train_linker_cfg_mgr.py
AUTHOR: RCB
CREATED: 2024-04-4-00:04
DESC:
"""
import json
import os.path
import shutil


class TrainCfgBase:
    __slots__ = [
        "file_name",
        "file_path",
        "cfg_obj",
    ]

    def __init__(self):
        self.file_name = "train_cfg.json"
        self.file_path = ""
        self.cfg_obj = {}

    def save_cfg(self, save_path):
        """
        保存配置文件
        @param path:
        @return:
        """

        if save_path.endswith(".json"):
            save_path = save_path
            with open(save_path, 'w') as f:
                json.dump(self.cfg_obj, f, indent=2)
            self.file_path = save_path
            self.file_name = os.path.basename(save_path)

    def load_cfg(self, path):
        """
        加载配置文件
        @param path:
        @return:
        """
        if os.path.exists(path) and os.path.isfile(path) and path.endswith(".json"):
            with open(path, 'r') as f:
                self.cfg_obj = json.load(f)
                self.file_name = os.path.basename(path)
                self.file_path = path

    @staticmethod
    def get_unique_filepath(filepath):
        # 如果文件不存在，直接返回原文件路径
        if not os.path.exists(filepath):
            return filepath

        # 如果文件存在，添加数字后缀
        base, extension = os.path.splitext(filepath)
        counter = 1

        while os.path.exists(filepath):
            filepath = f"{base}_{counter}{extension}"
            counter += 1

        return filepath

    def change_name(self, new_name):
        """
        修改配置文件名称
        @param new_name:
        @return:
        """
        if not new_name.endswith(".json"):
            new_name = new_name + ".json"
        new_path = os.path.join(os.path.dirname(self.file_path), new_name)
        new_path = self.get_unique_filepath(new_path)
        os.rename(self.file_path, new_path)
        self.file_name = new_name
        self.file_path = new_path

    def copy(self):
        """
        修改配置文件名称
        @param new_name:
        @return:
        """
        new_path = self.get_unique_filepath(self.file_path)

        shutil.copy(self.file_path, new_path)
        return new_path



class TrainCfgDreamBooth(TrainCfgBase):
    def __init__(self):
        super().__init__()
        self.cfg_obj = {
            "sdxl": True,
            "learning_rate": 2e-05,
            "learning_rate_te1": 2e-05,
            "learning_rate_te2": 2e-05,
            "train_text_encoder": True,
            "optimizer": "AdamW8bit",
            "max_train_epochs": 20,
            "save_every_n_epochs": 5,
            "sample_every_n_epochs": 5,
            "max_resolution": "1024,1024",
            "max_token_length": "75",
            "shuffle_caption": False,
            "min_snr_gamma": 5,
            "noise_offset_type": "Multires",
            "noise_offset": 0.03,
            "adaptive_noise_scale": 0.03,
            "multires_noise_iterations": 8,
            "multires_noise_discount": 0.3,
            "output_name": "last",
            "cus_lr_schedule": "1.0,1,0.9,1,0.8,1,0.7,1,0.6,1,0.5,1,0.4,1,0.3,1,0.2,1,0.1,1",
            "ch_debug_dataset": False,
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "full_16": True,
            "xformers": "sdpa",
            "debiased_estimation_loss": False,
            "gradient_accumulation_steps": "1",
            "logging_dir": "",
            "sample_tx_train_dir": "",
            "train_data_dir": "",
            "output_dir": "",
            "base_model": "",
            "vae": "",

        }


class TrainCfgLora(TrainCfgBase):
    def __init__(self):
        super().__init__()
        self.cfg_obj = {
            "sdxl": True,
            "learning_rate": 2e-05,
            "text_encoder_lr": 2e-05,
            "optimizer": "AdamW8bit",
            "max_train_epochs": 20,
            "save_every_n_epochs": 5,
            "sample_every_n_epochs": 5,
            "max_resolution": "1024,1024",
            "max_token_length": "75",
            "shuffle_caption": False,
            "min_snr_gamma": 5,
            "noise_offset_type": "Multires",
            "noise_offset": 0.03,
            "adaptive_noise_scale": 0.03,
            "multires_noise_iterations": 8,
            "multires_noise_discount": 0.3,
            "output_name": "last",
            "cus_lr_schedule": "1.0,1,0.9,1,0.8,1,0.7,1,0.6,1,0.5,1,0.4,1,0.3,1,0.2,1,0.1,1",
            "ch_debug_dataset": False,
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "full_16": True,
            "xformers": "sdpa",
            "debiased_estimation_loss": False,
            "gradient_accumulation_steps": "1",
            "logging_dir": "",
            "sample_tx_train_dir": "",
            "train_data_dir": "",
            "output_dir": "",
            "base_model": "",
            "vae": "",
            "network_dim": 128,
            "network_alpha": 32,
        }