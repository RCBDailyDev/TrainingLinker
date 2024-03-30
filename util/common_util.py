"""
FILE_NAME: common_util.py
AUTHOR: RCB
CREATED: 2024/3/30-15:09
DESC: 
"""
import json
import os
import subprocess
import json
import gradio as gr
from tkinter import filedialog, Tk

folder_symbol = '\U0001f4c2'  # 📂
load_symbol = '\U0001f4bf'  # 💿

img_ext_list = [".png", ".jpg", ".tga", ".bmp", ".webp"]
img_ext_tuple = (".png", ".jpg", ".tga", ".bmp", ".webp")
setting_path = os.path.expanduser('~\\AppData\\Local\\DataSetToolsConfig')

default_path = os.path.abspath(
    os.path.join(os.path.split(os.path.realpath(__file__))[0], os.path.pardir + "\\Temp_Menu"))


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


def open_folder(f, create=True):
    if f and not os.path.exists(f) or not os.path.isdir(f):
        if not create:
            return
        os.makedirs(f)
    path = os.path.normpath(f)
    subprocess.Popen(f'explorer /n,"{path}"')


def create_open_folder_button(path, elem_id, visible_in=True):
    button = gr.Button(value=folder_symbol, elem_id=elem_id, elem_classes="open_file_btn", visible=visible_in)
    if 'gradio.templates' in getattr(path, "__module__", ""):
        button.click(fn=lambda p: open_folder(p), inputs=[path], outputs=[])
    elif type(path) == gr.components.Textbox:
        button.click(fn=lambda p: open_folder(p), inputs=[path], outputs=[])
        pass
    else:
        button.click(fn=lambda: open_folder(path), inputs=[], outputs=[])
    return button


@Singleton
class ConfigMgr(object):
    def __init__(self):
        self.json_settings = {}
        # self.load_json_settings()

    def parse_json_data(self, main_key, json_data):
        self.json_settings[main_key] = {}
        for k in json_data:
            self.json_settings[main_key][k] = json_data[k]

    def load_json_settings(self, main_key):
        global setting_path
        os.makedirs(setting_path, exist_ok=True)
        json_path = os.path.join(setting_path, "{}.json".format(main_key))
        if not (os.path.exists(json_path) and os.path.isfile(json_path)):
            file = open(json_path, mode='w')
            file.close()
        with open(json_path, mode='r') as json_file:
            try:
                re = json.load(json_file)
                self.parse_json_data(main_key, re)
            except:
                self.json_settings[main_key] = {}

    def save_json_setting(self, main_key):
        json_path = os.path.join(setting_path, "{}.json".format(main_key))
        if not (os.path.exists(json_path) and os.path.isfile(json_path)):
            file = open(json_path, mode='w')
            file.close()
        with open(json_path, mode='w') as json_file:
            json.dump(self.json_settings[main_key], json_file)

    def get_cfg_value(self, main_key, key, default):
        if main_key in self.json_settings and key in self.json_settings[main_key]:
            return self.json_settings[main_key][key]
        return default

    def set_cfg_value(self, main_key, key, value):
        self.json_settings[main_key][key] = value


def get_cfg_mgr() -> ConfigMgr:
    return ConfigMgr()


@Singleton
class PathMgr:
    __slots__ = {
        "__train_cfg_path",
        "__prompt_lib_path",
    }

    def SetTrainCFGPath(self, path):
        self.__train_cfg_path = path

    def GetTrainCFGPath(self):
        return self.__train_cfg_path

    def SetPromptLibPath(self, path):
        self.__prompt_lib_path = path

    def GetPromptLibPath(self):
        return self.__prompt_lib_path


def get_path_mgr() -> PathMgr:
    return PathMgr()


def openfile_dialog(
        file_path, default_extension='.json', extension_name='Config files'
):
    out_path = ""
    if (not any(var in os.environ for var in ['COLAB_GPU', 'RUNPOD_POD_ID'])):
        # Create a hidden Tkinter root window
        root = Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()

        # Show the open file dialog and get the selected file path
        out_path = filedialog.askopenfilename(
            filetypes=(
                (extension_name, f'*{default_extension}'),
                ('All files', '*.*'),
            ),
            defaultextension=default_extension,
            initialdir=file_path,
        )

        root.destroy()

    return out_path
