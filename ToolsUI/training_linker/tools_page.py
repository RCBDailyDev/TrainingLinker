"""
FILE_NAME: tools_page.py
AUTHOR: RCB
CREATED: 2024/6/11-12:16
DESC: 
"""
import os.path

import gradio as gr
from ToolsUI.tab_base import TabBase
import ToolsUI.tab_mgr as tm
import util.common_util as util
import ToolsUI.training_linker.train_linker_cmd_maker as cmd_maker
import ToolsUI.training_linker.train_linker_excecutor as train_linker_executor


class ToolsPage(TabBase):
    def __init__(self):
        super().__init__()
        self.ui_kv = {}
        self.tab_name = "ToolsPage"
        self.cmd_executor = train_linker_executor.CommandExecutor()
        self.create_tab_ui()

        self.impl_change_save_logic(self.tab_name)
        tm.get_tab_mgr().registerTab(self)

    def create_tab_ui(self):
        cfg_mgr = util.get_cfg_mgr()
        cfg_mgr.load_json_settings(self.tab_name)
        with gr.Group():
            gr.HTML("<h2>Extract Lora From Model</h2>")
            with gr.Row():
                self.exl_tx_base_model = gr.Textbox(label="Base Model", value=cfg_mgr.get_cfg_value(self.tab_name,
                                                                                                    "exl_tx_base_model",
                                                                                                    ""))
                self.register_change_save("exl_tx_base_model", self.exl_tx_base_model)
                self.__RegisterUI("exl_tx_base_model", self.exl_tx_base_model)
                self.exl_btn_open_choose_base_model = gr.Button(value="\U0001f4c2", elem_classes="tl_set_small")

                self.exl_tx_tuned_model = gr.Textbox(label="Tuned Model", value=cfg_mgr.get_cfg_value(self.tab_name,
                                                                                                      "exl_tx_tuned_model",
                                                                                                      ""))
                self.register_change_save("exl_tx_tuned_model", self.exl_tx_tuned_model)
                self.__RegisterUI("exl_tx_tuned_model", self.exl_tx_tuned_model)
                self.exl_btn_open_choose_tuned_model = gr.Button(value="\U0001f4c2", elem_classes="tl_set_small")

            with gr.Row():
                self.exl_model_type = gr.Radio(value="SDXL", label="Model Type", choices=["SDXL", "V2", "V1"],
                                               interactive=True)
                self.__RegisterUI("exl_model_type", self.exl_model_type)
                self.exl_dim = gr.Radio(label="Dim", choices=[4, 8, 16, 32, 64, 128, 256, 512, 768, 1024],
                                        value=128, type="value", interactive=True)
                self.__RegisterUI("exl_dim", self.exl_dim)
                self.exl_device = gr.Radio(label="Device", choices=["cpu", "cuda"],
                                           value="cuda", type="value")
                self.__RegisterUI("exl_device", self.exl_device)
            with gr.Row():
                self.exl_load_precision = gr.Radio(label="LoadPrecision", value="fp16",
                                                   choices=["None", "float", "fp16", "bf16"])
                self.__RegisterUI("exl_load_precision", self.exl_load_precision)
                self.exl_save_precision = gr.Radio(label="SavePrecision", value="fp16",
                                                   choices=["None", "float", "fp16", "bf16"])
                self.__RegisterUI("exl_save_precision", self.exl_save_precision)
            with gr.Row():
                self.exl_tx_output_dir = gr.Textbox(label="OutputDir", value=cfg_mgr.get_cfg_value(self.tab_name,
                                                                                                   "exl_tx_output_dir",
                                                                                                   ""))
                self.register_change_save("exl_tx_output_dir", self.exl_tx_output_dir)
                self.__RegisterUI("exl_tx_output_dir", self.exl_tx_output_dir)
                util.create_open_folder_button(self.exl_tx_output_dir, "btn_open_exl_tx_output_dir")

                self.exl_tx_save_name = gr.Textbox(label="SaveName")
                self.register_change_save("exl_tx_save_name", self.exl_tx_save_name)
                self.__RegisterUI("exl_tx_save_name", self.exl_tx_save_name)

                self.exl_ch_overwrite = gr.Checkbox(label="Overwrite", value=False, interactive=True)
                self.__RegisterUI("exl_ch_overwrite", self.exl_ch_overwrite)

            with gr.Row():
                self.exl_btn_run = gr.Button(value="Run", elem_classes="tl_btn_common_green")
                self.exl_btn_stop = gr.Button(value="Stop", elem_classes="tl_btn_common_red")
            with gr.Row():
                self.ht_hint = gr.HTML()

        pass

    def impl_ui_logic(self):
        def exl_btn_open_choose_base_model_click(orig_path):
            fpath = util.openfile_dialog(orig_path, default_extension=".safetensors",
                                         extension_name="safetensors")
            if fpath:
                return fpath.replace("/", "\\")

        self.exl_btn_open_choose_base_model.click(fn=exl_btn_open_choose_base_model_click,
                                                  inputs=[self.exl_tx_base_model],
                                                  outputs=[self.exl_tx_base_model], show_progress="hidden")

        self.exl_btn_open_choose_tuned_model.click(fn=exl_btn_open_choose_base_model_click,
                                                   inputs=[self.exl_tx_tuned_model],
                                                   outputs=[self.exl_tx_tuned_model], show_progress="hidden")

        def exl_btn_run_click(*args):

            ui_list = list(self.ui_kv.keys())

            parm_kv = list(zip(ui_list, args))
            parm_dic = {}
            for k, v in parm_kv:
                parm_dic[k] = v
            base_model = parm_dic["exl_tx_base_model"]
            tuned_model = parm_dic["exl_tx_tuned_model"]
            if not base_model or not tuned_model:
                print("No Model Selected")
                return "<h3 class='tl_red_text'>Error 源模型路径为空</h3>"

            if not os.path.isfile(base_model) or not os.path.isfile(tuned_model):
                print("Not a Model File")
                return "<h3 class='tl_red_text'>Error 源模型不是模型文件</h3>"

            if not base_model.endswith("safetensors") or not tuned_model.endswith("safetensors"):
                print("Not a Model File")
                return "<h3 class='tl_red_text'>Error 源模型不是模型文件</h3>"

            if not parm_dic["exl_tx_output_dir"] or not os.path.isdir(parm_dic["exl_tx_output_dir"]):
                print("Output Dir Error")
                return "<h3 class='tl_red_text'>Error 输出路径错误</h3>"

            if not parm_dic["exl_tx_save_name"]:
                save_name = (os.path.basename(tuned_model).split('.')[0] + "-"
                             + os.path.basename(base_model).split('.')[0]) + ".safetensors"
            else:
                save_name = parm_dic["exl_tx_save_name"] + ".safetensors"

            output_file = os.path.join(parm_dic["exl_tx_output_dir"], save_name)
            if os.path.exists(output_file):
                if not parm_dic["exl_ch_overwrite"]:
                    print("File Exists")
                    return "<h3 class='tl_red_text'>Error 文件已存在</h3>"
            cmd_obj = {}
            cmd_obj["model_org"] = base_model
            cmd_obj["model_tuned"] = tuned_model
            cmd_obj["dim"] = int(parm_dic["exl_dim"])
            cmd_obj["save_precision"] = (parm_dic["exl_save_precision"])
            cmd_obj["load_precision"] = (parm_dic["exl_load_precision"])
            cmd_obj["device"] = parm_dic["exl_device"]
            cmd_obj["save_to"] = output_file
            cmd_obj["model_type"] = parm_dic["exl_model_type"]

            cmd = cmd_maker.make_extract_lora_from_model_cmd(cmd_obj)
            
            self.cmd_executor.execute_command(cmd)
            return "<h3 class='tl_green_text'>运行中，请看控制台</h3>"

        self.exl_btn_run.click(fn=exl_btn_run_click, inputs=self.__GetAllUI(), outputs=[self.ht_hint])
        self.exl_btn_stop.click(fn=lambda: self.cmd_executor.kill_command())

    def __RegisterUI(self, ui_key, ui):
        self.ui_kv[ui_key] = ui

    def __GetAllUI(self):
        return list(self.ui_kv.values())
