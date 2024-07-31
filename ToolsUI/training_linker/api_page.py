import datetime
import json
import math
import os.path

import gradio as gr
from ToolsUI.tab_base import TabBase
import ToolsUI.tab_mgr as tm
import ToolsUI.training_linker.train_linker_cmd_maker as data_set_cmd_maker
import ToolsUI.training_linker.train_linker_excecutor as train_linker_executor


class ApiPage(TabBase):
    def __init__(self):
        super().__init__()
        self.tab_name = "ApiPage"
        self.cmd_executor = train_linker_executor.CommandExecutor()
        self.create_tab_ui()
        self.impl_change_save_logic(self.tab_name)
        tm.get_tab_mgr().registerTab(self)

    def create_tab_ui(self):
        with gr.Group():
            gr.HTML("<h1>API Page Nothing To Show</h1>")
            pass
        self.json_cmd_json = gr.Json(label="CmdJson", visible=False)
        self.btn_run_db = gr.Button(visible=False)
        self.btn_run_lora = gr.Button(visible=False)
        self.btn_stop_db = gr.Button(visible=False)
        self.btn_stop_lora = gr.Button(visible=False)
        self.btn_print_lora_cmd = gr.Button(visible=False)

    def impl_ui_logic(self):
        def btn_run_api_click(cfg_obj):
            self.cmd_executor.execute_command(data_set_cmd_maker.make_db_cmd(cfg_obj))

        self.btn_run_db.click(fn=btn_run_api_click, inputs=[self.json_cmd_json], api_name="train_db")

        self.btn_stop_db.click(fn=lambda: self.cmd_executor.kill_command(), api_name="stop_db")

        def btn_run_api_click(cfg_obj):
            self.cmd_executor.execute_command(data_set_cmd_maker.make_lora_cmd(cfg_obj))

        self.btn_run_lora.click(fn=btn_run_api_click, inputs=[self.json_cmd_json], api_name="train_lora")

        self.btn_stop_lora.click(fn=lambda: self.cmd_executor.kill_command(), api_name="stop_lora")

        def btn_print_lora_cmd_click(cfg_obj):
            print("=====================CmdPreview=======================")
            print(data_set_cmd_maker.make_lora_cmd(cfg_obj))
            print("======================================================")

        self.btn_print_lora_cmd.click(fn=btn_print_lora_cmd_click, inputs=[self.json_cmd_json],
                                      api_name="print_lora_cmd")
