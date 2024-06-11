"""
FILE_NAME: tab_base.py
AUTHOR: RCB
CREATED: 2024/1/2-14:43
DESC: tab 页面基类
"""
from abc import ABC, abstractmethod
import gradio as gr
import util.common_util as util


class TabBase(ABC):
    def __init__(self):
        self.change_save_list = []
        self.tab_name = None
    @abstractmethod
    def create_tab_ui(self):
        pass

    @abstractmethod
    def impl_ui_logic(self):
        pass

    def register_change_save(self, key, ui):
        if (key, ui) not in self.change_save_list:
            self.change_save_list.append((key, ui))

    def impl_change_save_logic(self, main_key):
        def save_cfg_on_change(*args):
            cfg_mgr = util.get_cfg_mgr()
            for idx,arg in enumerate(args):
                cfg_mgr.set_cfg_value(main_key, self.change_save_list[idx][0], arg)
            cfg_mgr.save_json_setting(main_key)

        for ui in self.change_save_list:
            if type(ui[1]) == gr.components.Textbox:
                ui[1].blur(fn=save_cfg_on_change, inputs=[x[1] for x in self.change_save_list], outputs=None)
                ui[1].change(fn=save_cfg_on_change, inputs=[x[1] for x in self.change_save_list], outputs=None)
            else:
                ui[1].change(fn=save_cfg_on_change, inputs=[x[1] for x in self.change_save_list], outputs=None)
