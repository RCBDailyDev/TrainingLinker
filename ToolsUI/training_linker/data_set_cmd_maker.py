"""
FILE_NAME: data_set_cmd_maker.py
AUTHOR: RCB
CREATED: 2024/4/2-11:54
DESC: 
"""
import os


def make_db_cmd(cfg_obj):
    cmd = "accelerate launch"
    cmd += f" --num_cpu_threads_per_process={2}"
    if cfg_obj["sdxl"]:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "sdxl_train.py") + "\""
    else:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "train_db.py") + "\""
    return cmd
