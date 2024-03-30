import datetime
import json
import math
import os.path

import gradio as gr
import util.common_util as util
from ToolsUI import data_set_mgr_constant as dsmc
from gradio_client import Client
from ToolsUI.tab_base import TabBase
import ToolsUI.tab_mgr as tm


class DataSetTrainingLinker(TabBase):
    def __init__(self, standalone=False):
        super().__init__()
        self.root_path = ""
        self.tab_name = "DataSetTrainingLinker"
        self.standalone = standalone
        self.ui_kv = {}
        self.create_tab_ui()
        self.impl_change_save_logic(self.tab_name)
        self.__ImplConfigUpdate()
        tm.get_tab_mgr().registerTab(self)

    def create_tab_ui(self):
        with gr.Group():
            with gr.Row():
                self.dd_cfg_list = gr.Dropdown(label=dsmc.TRAIN_CFG, choices=[], interactive=True)
                self.btn_reload_cfg = gr.Button(value="\U0001f504", elem_classes="res_set_small")
                self.btn_open_cfg_path = gr.Button(value="\U0001f4c2", elem_classes="res_set_small")
                self.btn_load_cfg = gr.Button(value="LoadCfg")

            with gr.Row() as self.row_detail:
                self.row_detail.visible = False
                with gr.Column():
                    with gr.Row():
                        self.sdxl = gr.Checkbox(
                            label='SDXL', value=True, interactive=True
                        )
                        self.__RegisterUI("sdxl", self.sdxl)

                        self.learning_rate = gr.Number(
                            label='Learning rate', value=1e-6, interactive=True
                        )
                        self.__RegisterUI("learning_rate", self.learning_rate)
                    with gr.Row():
                        self.train_text_encoder = gr.Checkbox(label="train_text_encoder", value=True, interactive=True)
                        self.__RegisterUI("train_text_encoder", self.train_text_encoder)
                        self.learning_rate_te1 = gr.Number(label="learning_rate_te1", value=1e-6, interactive=True)
                        self.__RegisterUI("learning_rate_te1", self.learning_rate_te1)
                        self.learning_rate_te2 = gr.Number(label="learning_rate_te2", value=1e-6, interactive=True)
                        self.__RegisterUI("learning_rate_te2", self.learning_rate_te2)
                    with gr.Row():
                        self.optimizer = gr.Dropdown(
                            label='Optimizer',
                            choices=[
                                'AdamW8bit',
                                'Lion'
                            ],
                            value='AdamW8bit',
                            interactive=True,
                        )
                        self.__RegisterUI("optimizer", self.optimizer)
                    with gr.Row():
                        self.max_train_epochs = gr.Number(
                            label='Max train epoch',
                            value=10,
                            minimum=1,
                            precision=0,
                            interactive=True
                        )
                        self.__RegisterUI("max_train_epochs", self.max_train_epochs)
                        self.save_every_n_epochs = gr.Number(
                            label='Save every N epochs', value=5, precision=0, interactive=True
                        )
                        self.__RegisterUI("save_every_n_epochs", self.save_every_n_epochs)
                        self.sample_every_n_epochs = gr.Number(
                            label='Sample every N epochs', value=5, precision=0, interactive=True
                        )
                        self.__RegisterUI("sample_every_n_epochs", self.sample_every_n_epochs)
                        self.max_resolution = gr.Textbox(
                            label='Max resolution',
                            value='1024,1024',
                            placeholder='1024,1024',
                        )
                        self.__RegisterUI("max_resolution", self.max_resolution)
                    with gr.Row():
                        self.gradient_accumulation_steps = gr.Slider(
                            label='Gradient accumulate steps',
                            info='Number of updates steps to accumulate before performing a backward/update pass',
                            value='1',
                            minimum=1,
                            maximum=120,
                            step=1,
                            interactive=True
                        )
                        self.__RegisterUI("max_resolution", self.gradient_accumulation_steps)
                        self.max_token_length = gr.Dropdown(
                            label='Max Token Length',
                            choices=[
                                '75',
                                '150',
                                '225',
                            ],
                            value='75',
                            interactive=True
                        )
                        self.__RegisterUI("max_token_length", self.max_token_length)
                        self.shuffle_caption = gr.Checkbox(
                            label='Shuffle caption', value=False
                        )
                        self.__RegisterUI("shuffle_caption", self.shuffle_caption)
                        self.min_snr_gamma = gr.Slider(
                            label='Min SNR gamma',
                            value=5,
                            minimum=0,
                            maximum=20,
                            step=1,
                            info='Recommended value of 5 when used',
                            interactive=True
                        )
                        self.__RegisterUI("min_snr_gamma", self.min_snr_gamma)
                    with gr.Row():
                        self.noise_offset_type = gr.Dropdown(
                            label='Noise offset type',
                            choices=[
                                'Original',
                                'Multires',
                            ],
                            value='Multires',
                            interactive=True
                        )
                        self.__RegisterUI("noise_offset_type", self.noise_offset_type)
                        with gr.Row(visible=True) as self.noise_offset_original:
                            self.noise_offset = gr.Slider(
                                label='Noise offset',
                                value=0,
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                info='recommended values are 0.05 - 0.15',
                                interactive=True
                            )
                            self.__RegisterUI("noise_offset", self.noise_offset)
                            self.adaptive_noise_scale = gr.Slider(
                                label='Adaptive noise scale',
                                value=0,
                                minimum=-1,
                                maximum=1,
                                step=0.001,
                                info='(Experimental, Optional) Since the latent is close to a normal distribution, it may be a good idea to specify a value around 1/10 the noise offset.',
                                interactive=True
                            )
                            self.__RegisterUI("adaptive_noise_scale", self.adaptive_noise_scale)
                        with gr.Row(visible=False) as self.noise_offset_multires:
                            self.multires_noise_iterations = gr.Slider(
                                label='Multires noise iterations',
                                value=8,
                                minimum=0,
                                maximum=64,
                                step=1,
                                info='enable multires noise (recommended values are 6-10)',
                                interactive=True
                            )
                            self.__RegisterUI("multires_noise_iterations", self.multires_noise_iterations)
                            self.multires_noise_discount = gr.Slider(
                                label='Multires noise discount',
                                value=0.5,
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                info='recommended values are 0.8. For LoRAs with small datasets, 0.1-0.3',
                                interactive=True
                            )
                            self.__RegisterUI("multires_noise_discount", self.multires_noise_discount)
                    with gr.Row():
                        self.output_dir = gr.Textbox(label="OutputPath", interactive=True)
                        self.__RegisterUI("output_dir", self.output_dir)
                        util.create_open_folder_button(self.output_dir, "btn_open_output_dir")
                        self.output_name = gr.Textbox(label="SaveName", value="last", interactive=True)
                        self.__RegisterUI("output_name", self.output_name)
                    with gr.Row():
                        self.base_model = gr.Textbox(label="BaseModel", interactive=True)
                        self.__RegisterUI("base_model", self.base_model)
                        self.btn_open_source_model_choose = gr.Button(value="\U0001f4c2", elem_classes="res_set_small")
                        self.btn_choose_last = gr.Button(value="ChooseLastModel", elem_classes="dsm_btn_small_green")
                    with gr.Row():
                        self.vae = gr.Textbox(label="VaeModel", interactive=True)
                        self.__RegisterUI("vae", self.vae)
                        self.btn_open_vae_choose = gr.Button(value="\U0001f4c2", elem_classes='res_set_small')
                    with gr.Row():
                        self.cus_lr_schedule = gr.Textbox(label="LRSchedule", interactive=True)
                        self.__RegisterUI("cus_lr_schedule", self.cus_lr_schedule)
                    with gr.Row():
                        self.tx_train_ip = gr.Textbox(label="TrainIP", value="127.0.0.1")
                        self.__RegisterUI("tx_train_ip", self.tx_train_ip)
                        self.tx_train_port = gr.Textbox(label="TrainPort", value="7682", interactive=True)
                        self.__RegisterUI("tx_train_port", self.tx_train_port)
                    with gr.Row():
                        self.ch_debug_dataset = gr.Checkbox(label="DebugDataset", value=False, interactive=True)
                        self.__RegisterUI("ch_debug_dataset", self.ch_debug_dataset)
                    with gr.Accordion("Advance"):
                        with gr.Row():
                            self.mixed_precision = gr.Dropdown(
                                label='Mixed precision',
                                choices=[
                                    'no',
                                    'fp16',
                                    'bf16',
                                ],
                                value='fp16',
                                interactive=True,
                            )
                            self.__RegisterUI("mixed_precision", self.mixed_precision)
                            self.save_precision = gr.Dropdown(
                                label='Save precision',
                                choices=[
                                    'float',
                                    'fp16',
                                    'bf16',
                                ],
                                value='fp16',
                                interactive=True,
                            )
                            self.__RegisterUI("save_precision", self.save_precision)
                            self.full_16 = gr.Checkbox(
                                label='Full fp16 or bp16 training (experimental)',
                                value=False,
                                interactive=True,
                            )
                            self.__RegisterUI("full_16", self.full_16)
                        self.xformers = gr.Dropdown(
                            label='CrossAttention',
                            choices=['none', 'sdpa', 'xformers'],
                            value='xformers',
                            interactive=True,
                        )
                        self.__RegisterUI("xformers", self.xformers)
                        self.debiased_estimation_loss = gr.Checkbox(label="debiased_estimation_loss", value=False)
                        self.__RegisterUI("debiased_estimation_loss", self.debiased_estimation_loss)
                    with gr.Row():
                        self.btn_save = gr.Button(value="SaveCfg", elem_classes="dsm_btn_common_blue")
                        self.btn_run = gr.Button(value="Run", elem_classes="dsm_btn_common_green")
                        self.btn_stop = gr.Button(value="Stop", elem_classes="dsm_btn_common_red")
                    with gr.Row():
                        self.btn_get_lora_cfg = gr.Button(value="GetLoraCFG")
                    with gr.Group():
                        with gr.Row():
                            self.num_start_lr = gr.Number(label="Start Learning Rate", minimum=0, value=1,
                                                          interactive=True)
                            self.num_end_lr = gr.Number(label="End Learning", minimum=0, value=0.1, interactive=True)
                        with gr.Row():
                            self.num_warmup_epoch = gr.Number(label="Warmup Epoch", minimum=0, value=0,
                                                              interactive=True)
                            self.num_warmup_step = gr.Number(label="Warmup Step", minimum=1, value=1, interactive=True)
                            self.num_cool_step = gr.Number(label="Cool Step", minimum=1, value=1, interactive=True)
                        with gr.Row():
                            self.btn_gen_lr_schedule_list = gr.Button(value="GenSchedule")

    def impl_ui_logic(self):
        dataset_tab = tm.get_tab_mgr().getTab("DataSetMgrTab")

        def btn_reload_cfg_click():
            if self.standalone:
                root_path = "../../TrainDataPath"
            else:
                root_path = self.root_path
            return gr.update(choices=self.GetCfgList(root_path))

        self.btn_reload_cfg.click(fn=btn_reload_cfg_click, outputs=[self.dd_cfg_list])

        def btn_gen_lr_schedule_list_click(total_epoch, start_lr, end_lr, warmup_epoch, warmup_step, cool_step):
            ret_str = []
            for i in range(0, math.ceil(warmup_epoch / warmup_step)):
                time = i / (math.ceil(warmup_epoch / warmup_step) - 1)
                lr = start_lr * (1 - time) + time
                ret_str.append(str(round(lr, 3)))
                ret_str.append(str(warmup_step))
            for i in range(0, math.ceil((total_epoch - warmup_epoch) / cool_step)):
                time = i / (math.ceil((total_epoch - warmup_epoch) / cool_step) - 1)
                lr = (1 - time) + end_lr * time
                ret_str.append(str(round(lr, 3)))
                ret_str.append(str(cool_step))
            return ",".join(ret_str)

        self.btn_gen_lr_schedule_list.click(fn=btn_gen_lr_schedule_list_click,
                                            inputs=[self.max_train_epochs, self.num_start_lr, self.num_end_lr,
                                                    self.num_warmup_epoch,
                                                    self.num_warmup_step, self.num_cool_step],
                                            outputs=[self.cus_lr_schedule])

        def btn_get_lora_cfg_click():
            self.__CheckLoraCfgPath()
            cfg_list = self.__GetLoraCfgList()
            cfg_path = os.path.join(self.root_path, "TrainLoraCfg", cfg_list[0])
            # with open(cfg_path, 'r') as temp:
            #     template_obj = json.load(temp)
            import pyperclip
            pyperclip.copy(cfg_path)

        self.btn_get_lora_cfg.click(fn=btn_get_lora_cfg_click)

        def btn_load_cfg_click(cfg_name):
            ui_count = len(self.__GetUIList())
            skip_list = [gr.skip()] * ui_count
            if not cfg_name:
                return [gr.update(visible=False)] + skip_list
            cfg_file_path = os.path.join(self.root_path, dsmc.TRAIN_CFG, cfg_name)
            if not os.path.exists(cfg_file_path) or not os.path.isfile(cfg_file_path):
                return [gr.update(visible=False)] + skip_list
            self.__LoadCfg(cfg_file_path)

            ui_state_list = []
            for k, _v in self.ui_kv.items():
                if k in self.cfg_obj:
                    ui_state_list.append(gr.update(value=self.cfg_obj[k]))
                else:
                    ui_state_list.append(gr.skip())
            if "noise_offset_type" in self.cfg_obj:
                if self.cfg_obj["noise_offset_type"] == "Original":
                    ui_state_list.append(gr.update(visible=True))
                    ui_state_list.append(gr.update(visible=False))
                else:
                    ui_state_list.append(gr.update(visible=False))
                    ui_state_list.append(gr.update(visible=True))
            else:
                ui_state_list.append(gr.update(visible=False))
                ui_state_list.append(gr.update(visible=True))
            return [gr.update(visible=True)] + ui_state_list

        self.btn_load_cfg.click(fn=btn_load_cfg_click, inputs=[self.dd_cfg_list],
                                outputs=[self.row_detail] + self.__GetUIList())

        def btn_save_click(cfg_name):
            cfg_file_path = os.path.join(self.root_path, dsmc.TRAIN_CFG, cfg_name)
            self.__SaveCfg(cfg_file_path)
            ##TODO:DELETE
            print("SaveCFG!")

        self.btn_save.click(fn=btn_save_click, inputs=[self.dd_cfg_list])

        def btn_open_source_model_choose_click():
            fpath = util.openfile_dialog(self.cfg_obj["output_dir"], default_extension=".safetensors",
                                         extension_name="safetensors")
            if fpath:
                return fpath.replace("/", "\\")

            return gr.skip()

        self.btn_open_source_model_choose.click(fn=btn_open_source_model_choose_click,
                                                outputs=[self.base_model], show_progress="hidden")

        def btn_open_vae_choose_click():
            fpath = util.openfile_dialog(self.cfg_obj['vae'], default_extension=".safetensors",
                                         extension_name="safetensors")
            if fpath:
                return fpath.replace("/", "\\")

            return gr.skip()

        self.btn_open_vae_choose.click(fn=btn_open_vae_choose_click, outputs=[self.vae], show_progress="hidden")

        def btn_choose_last_click():
            # find all ckpt
            output_path = self.cfg_obj["output_dir"]
            if not os.path.exists(output_path) or not os.path.isdir(output_path):
                return gr.skip()
            ckpt_list = []
            for r, d, fs in os.walk(output_path):
                for f in fs:
                    if f.endswith(".safetensors"):
                        file_path = os.path.join(r, f)
                        timestamp = os.path.getmtime(file_path)
                        ckpt_list.append((file_path, timestamp))
            if len(ckpt_list) <= 0:
                return gr.skip()
            ckpt_list.sort(key=lambda x: x[1], reverse=True)

            return ckpt_list[0][0]

        self.btn_choose_last.click(fn=btn_choose_last_click, outputs=[self.base_model])

        def btn_train_click(ip, port, debug_dataset):

            path_mgr = util.get_path_mgr()
            #  os.path.join(path_mgr.GetTrainCFGPath(), "Label_False.json"),
            #  os.path.join(path_mgr.GetTrainCFGPath(), "Label_True.json"),

            # deal output dir
            output_dir_root = self.cfg_obj['output_dir']
            if not output_dir_root:
                print("Output Dir Not Available")
                return
            if not os.path.exists(output_dir_root) or not os.path.isdir(output_dir_root):
                print("Output Dir Not Available")
                return
            # get time string
            import time
            timestamp = time.time()
            time_tuple = time.localtime(timestamp)
            formatted_time = time.strftime("%m-%d-%H-%M-%S", time_tuple)

            final_output_dir = os.path.join(output_dir_root, formatted_time)
            os.makedirs(final_output_dir, exist_ok=True)
            print("Output Model to: ", final_output_dir)
            self.__SaveTrainTrainInfo(final_output_dir, formatted_time)

            client = Client("http://{}:{}/".format(ip, port))
            result = client.predict(
                {"label": "False"},
                {"label": "False"},
                self.cfg_obj['pretrained_model_name_or_path'],
                self.cfg_obj['v2'],
                self.cfg_obj['v_parameterization'],
                self.cfg_obj['sdxl'],
                self.cfg_obj['logging_dir'],
                self.cfg_obj['train_data_dir'],
                self.cfg_obj['reg_data_dir'],
                final_output_dir,  # self.cfg_obj['output_dir'],
                self.cfg_obj['max_resolution'],
                self.cfg_obj['learning_rate'],
                self.cfg_obj['learning_rate_te1'],
                self.cfg_obj['learning_rate_te1'],
                self.cfg_obj['learning_rate_te2'],
                self.cfg_obj['lr_scheduler'],
                self.cfg_obj['lr_warmup'],
                self.cfg_obj['train_batch_size'],
                self.cfg_obj['epoch'],
                self.cfg_obj['save_every_n_epochs'],
                self.cfg_obj['mixed_precision'],
                self.cfg_obj['save_precision'],
                self.cfg_obj['seed'],
                self.cfg_obj['num_cpu_threads_per_process'],
                self.cfg_obj['cache_latents'],
                self.cfg_obj['cache_latents_to_disk'],
                self.cfg_obj['caption_extension'],
                self.cfg_obj['enable_bucket'],
                self.cfg_obj['gradient_checkpointing'],
                self.cfg_obj['full_fp16'],
                self.cfg_obj['full_bf16'],
                self.cfg_obj['no_token_padding'],
                0,
                self.cfg_obj['min_bucket_reso'],
                self.cfg_obj['max_bucket_reso'],
                self.cfg_obj['xformers'],
                self.cfg_obj['save_model_as'],
                self.cfg_obj['shuffle_caption'],
                self.cfg_obj['save_state'],
                self.cfg_obj['resume'],
                self.cfg_obj['prior_loss_weight'],
                self.cfg_obj['color_aug'],
                self.cfg_obj['flip_aug'],
                self.cfg_obj['clip_skip'],
                self.cfg_obj['vae'],
                1,
                1,
                False,
                "",
                self.cfg_obj['output_name'],
                self.cfg_obj['max_token_length'],
                self.cfg_obj['max_train_epochs'],
                self.cfg_obj['max_train_steps'],
                self.cfg_obj['max_data_loader_n_workers'],
                self.cfg_obj['mem_eff_attn'],
                self.cfg_obj['gradient_accumulation_steps'],
                self.cfg_obj['model_list'],
                self.cfg_obj['keep_tokens'],
                self.cfg_obj['lr_scheduler_num_cycles'],
                self.cfg_obj['lr_scheduler_power'],
                self.cfg_obj['persistent_data_loader_workers'],
                self.cfg_obj['bucket_no_upscale'],
                self.cfg_obj['random_crop'],
                self.cfg_obj['bucket_reso_steps'],
                self.cfg_obj['v_pred_like_loss'],
                self.cfg_obj['caption_dropout_every_n_epochs'],
                self.cfg_obj['caption_dropout_rate'],
                self.cfg_obj['optimizer'],
                self.cfg_obj['optimizer_args'],
                self.cfg_obj['lr_scheduler_args'],
                self.cfg_obj['noise_offset_type'],
                self.cfg_obj['noise_offset'],
                self.cfg_obj['adaptive_noise_scale'],
                self.cfg_obj['multires_noise_iterations'],
                self.cfg_obj['multires_noise_discount'],
                self.cfg_obj['sample_every_n_steps'],
                self.cfg_obj['sample_every_n_epochs'],
                self.cfg_obj['sample_sampler'],
                self.cfg_obj['sample_prompts'],
                self.cfg_obj['additional_parameters'] + "--debug_dataset" if debug_dataset else "",
                self.cfg_obj['vae_batch_size'],
                self.cfg_obj['min_snr_gamma'],
                self.cfg_obj['weighted_captions'],
                self.cfg_obj['save_every_n_steps'],
                self.cfg_obj['save_last_n_steps'],
                self.cfg_obj['save_last_n_steps_state'],
                self.cfg_obj['use_wandb'],
                self.cfg_obj['wandb_api_key'],
                self.cfg_obj['scale_v_pred_loss_like_noise_pred'],
                self.cfg_obj['min_timestep'],
                self.cfg_obj['max_timestep'],
                self.cfg_obj['sample_tx_train_dir'],
                self.cfg_obj['cus_num_drop_tag_epoch'],
                self.cfg_obj['cus_sl_tag_drop_rate'],
                self.cfg_obj['cus_tx_keep_tags'],
                self.cfg_obj['cus_num_keep_tag_count'],
                self.cfg_obj['cus_ch_adv_tag_dropout'],
                self.cfg_obj['cus_lr_schedule'],

                api_name="/train_db"
            )
            ##TODO:DELETE
            print(result)

        self.btn_run.click(fn=btn_train_click, inputs=[self.tx_train_ip, self.tx_train_port, self.ch_debug_dataset])

        def btn_stop_click(ip, port):
            client = Client("http://{}:{}/".format(ip, port))
            result = client.predict(
                api_name="/stop_db"
            )

        self.btn_stop.click(fn=btn_stop_click, inputs=[self.tx_train_ip, self.tx_train_port])

        self.btn_open_cfg_path.click(lambda: util.open_folder(os.path.join(self.root_path, dsmc.TRAIN_CFG), False))

    def __ImplConfigUpdate(self):
        def set_value(obj, k, v):
            obj[k] = v

        for key, ui in self.ui_kv.items():
            if key == "noise_offset_type":
                def noise_offset_type_change(noise_offset_type):
                    self.cfg_obj[key] = noise_offset_type
                    if noise_offset_type == 'Original':
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                        )
                    else:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=True),
                        )
                ui.change(
                    noise_offset_type_change,
                    inputs=[ui],
                    outputs=[
                        self.noise_offset_original,
                        self.noise_offset_multires,
                    ],
                )
            else:
                ui.change(fn=lambda v: set_value(self.cfg_obj, key, v), inputs=[ui])

    def __RegisterUI(self, key, ui):
        if (key, ui) not in self.ui_kv:
            self.ui_kv[key] = ui
        else:
            print("Key {} already exists!".format(key))

    def GetCfgList(self, root_path):
        self.root_path = root_path
        cfg_path = os.path.join(root_path, dsmc.TRAIN_CFG)
        if not os.path.exists(cfg_path):
            os.makedirs(cfg_path, exist_ok=True)
        ret_list = []
        dir_l = os.listdir(cfg_path)
        for d in dir_l:
            if os.path.isfile(os.path.join(cfg_path, d)):
                if d.endswith(".json"):
                    ret_list.append(d)
        if len(ret_list) <= 0:
            self.__CreateDefaultCfg()
            ret_list.append("DefaultDbCfg.json")
        return ret_list

    def __LoadCfg(self, path):
        with open(path, 'r') as temp:
            self.cfg_obj = json.load(temp)
            if "tx_train_port" not in self.cfg_obj:
                self.cfg_obj["tx_train_port"] = "7862"
            if "tx_train_ip" not in self.cfg_obj:
                self.cfg_obj["tx_train_ip"] = "127.0.0.1"
            self.cfg_obj["logging_dir"] = os.path.join(self.root_path, dsmc.LOG_PATH)
            self.cfg_obj["sample_tx_train_dir"] = os.path.join(self.root_path, dsmc.SAM_DATA_PATH)
            self.cfg_obj["train_data_dir"] = os.path.join(self.root_path, dsmc.TRAIN_PATH)

    def __GetUIList(self):
        return list(self.ui_kv.values()) + [self.noise_offset_original, self.noise_offset_multires]

    def __SaveCfg(self, path):
        with open(path, 'w') as file:
            json.dump(self.cfg_obj, file, indent=2)

    # region Local Method
    def __CheckCfgPath(self):
        # Check Folder
        cfg_path = os.path.join(self.root_path, dsmc.TRAIN_CFG)
        os.makedirs(cfg_path, exist_ok=True)

    def __CheckLoraCfgPath(self):
        # Check Folder
        cfg_path = os.path.join(self.root_path, "TrainLoraCfg")
        os.makedirs(cfg_path, exist_ok=True)

    def __SaveTrainTrainInfo(self, save_path, time_str):
        source_model = self.cfg_obj["pretrained_model_name_or_path"]
        info_obj = {}
        if source_model and os.path.exists(source_model):
            info_obj["base_model"] = os.path.basename(source_model)
            base_mode_time = os.path.getmtime(source_model)
            modified_time = datetime.datetime.fromtimestamp(base_mode_time)
            formatted_time = modified_time.strftime('%Y-%m-%d %H:%M:%S')
            info_obj["base_model_date"] = formatted_time
        train_img_path = self.cfg_obj['train_data_dir']
        if train_img_path and os.path.exists(train_img_path):
            prompt_stat = {}
            img_count = 0
            for r, d, fs in os.walk(train_img_path):
                for f in fs:
                    if f.endswith(".txt"):
                        with open(os.path.join(r, f), 'r') as file:
                            str = file.read()
                        p_list = str.split(',')
                        for p in p_list:
                            p = p.strip()
                            if p in prompt_stat:
                                prompt_stat[p] += 1
                            else:
                                prompt_stat[p] = 1
                        pass
                    if f.endswith((".png", ".jpg", ".tga", ".webp")):
                        img_count += 1
            p_stat_list = list(prompt_stat.items())
            p_stat_list.sort(key=lambda x: x[1], reverse=True)
            info_obj["img_count"] = img_count
            info_obj["p_stat"] = p_stat_list
        save_file = os.path.join(save_path, "TrainInfo_" + time_str + ".json")
        with open(save_file, 'w') as f:
            json.dump(info_obj, f, indent=2)

    def __GetCfgList(self):
        cfg_path = os.path.join(self.root_path, dsmc.TRAIN_CFG)
        ret_list = []
        dir_l = os.listdir(cfg_path)
        for d in dir_l:
            if os.path.isfile(os.path.join(cfg_path, d)):
                if d.endswith(".json"):
                    ret_list.append(d)
        if len(ret_list) <= 0:
            self.__CreateDefaultCfg()
            ret_list.append("Default.json")
        return ret_list

    def __GetLoraCfgList(self):
        cfg_path = os.path.join(self.root_path, dsmc.LORA_CFG)
        ret_list = []
        dir_l = os.listdir(cfg_path)
        for d in dir_l:
            if os.path.isfile(os.path.join(cfg_path, d)):
                if d.endswith(".json"):
                    ret_list.append(d)
        if len(ret_list) <= 0:
            self.__CreateDefaultCfg("DefaultLora", dsmc.LORA_CFG, "DefaultLoraCfg.json")
            ret_list.append("DefaultLora.json")
        return ret_list

    def __CreateDefaultCfg(self, file_name_in="Default", default_path_name=dsmc.TRAIN_CFG,
                           default_file_name="DefaultDbCfg.json"):
        cfg_path = os.path.join(self.root_path, default_path_name)
        file_name = file_name_in + ".json"
        save_path = os.path.join(cfg_path, file_name)
        path_mgr = util.get_path_mgr()
        template_path = os.path.join(path_mgr.GetTrainCFGPath(), default_file_name)
        with open(template_path, 'r') as temp:
            template_obj = json.load(temp)
        template_obj["logging_dir"] = os.path.join(self.root_path, dsmc.LOG_PATH)
        template_obj["sample_tx_train_dir"] = os.path.join(self.root_path, dsmc.SAM_DATA_PATH)
        template_obj["train_data_dir"] = os.path.join(self.root_path, dsmc.TRAIN_PATH)
        with open(save_path, 'w') as file:
            json.dump(template_obj, file, indent=2)

    # endregion
