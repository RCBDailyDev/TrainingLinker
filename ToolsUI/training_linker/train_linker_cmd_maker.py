"""
FILE_NAME: data_set_cmd_maker.py
AUTHOR: RCB
CREATED: 2024/4/2-11:54
DESC: 
"""
import json
import os
from PIL import Image
import datetime


def parse_param(s):
    p_list = s.split(',')
    if len(p_list) == 2:
        return int(p_list[0]), int(p_list[1])
    return 768, 768


def find_image(txt_path):
    img_ext = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    path_name = os.path.splitext(txt_path)[0]
    for ext in img_ext:
        img_path = path_name + ext
        if os.path.exists(img_path) and os.path.isfile(img_path):
            return img_path
    return None


def make_prompt_file(sample_path):
    info_list = []
    sample_prompts_path = sample_path + "\\sample\\prompt.json"
    negative_prompt = ""
    if sample_path:
        path_list = []
        sub_dir = os.listdir(sample_path)
        for sub in sub_dir:
            if sub != "sample":
                p = os.path.join(sample_path, sub)
                for r, d, fs in os.walk(p):
                    for f in fs:
                        if f.endswith(".txt"):
                            if f != "negative.txt":
                                path_list.append(os.path.join(r, f))
                            else:
                                with open(os.path.join(r, f), 'r') as txtf:
                                    negative_prompt = txtf.read()

        for txt_path in path_list:
            if not txt_path:
                continue
            img_path = find_image(txt_path)
            if img_path:
                img = Image.open(img_path)
                info_dic = {'width': img.width, 'height': img.height, 'scale': 7, 'negative_prompt': negative_prompt,
                            'sample_steps': 20}
                if not os.path.exists(txt_path):
                    raise ValueError("img_path has np caption")
                with open(txt_path, 'r') as txtf:
                    info_dic["prompt"] = txtf.read()
                info_dic['img_path'] = img_path
                info_list.append(info_dic)
            else:
                with open(txt_path, 'r') as txtf:
                    lines = txtf.readlines()
                if len(lines) > 0 and len(lines) % 2 == 0:
                    for i in range(0, len(lines), 2):
                        con = lines[i].strip()
                        parm = lines[i + 1].strip()
                        (w, h) = parse_param(parm)
                        info_dic = {'width': w, 'height': h, 'scale': 7, 'negative_prompt': negative_prompt,
                                    'sample_steps': 20, "prompt": con}
                        info_list.append(info_dic)
        txt = ""
        for idx, info in enumerate(info_list):
            txt += "{}: {}\n".format(idx, info["prompt"])
            sample_prompt = os.path.join(sample_path, "sample", 'prompt.txt')
            with open(sample_prompt, 'w') as fp:
                fp.write(txt)
        with open(sample_prompts_path, 'w') as f:
            json.dump(info_list, f)


def make_db_cmd(cfg_obj):
    cmd = "accelerate launch"
    cmd += f" --num_cpu_threads_per_process={2}"
    if cfg_obj["sdxl"]:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "sdxl_train.py") + "\""
    else:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "train_db.py") + "\""

    cmd += f" --bucket_no_upscale"
    cmd += f" --bucket_reso_steps={256}"
    cmd += f" --cache_latents"
    cmd += f" --caption_extension=\".txt\""
    cmd += f" --enable_bucket"
    cmd += f" --min_bucket_reso={768}"
    cmd += f" --max_bucket_reso={1024}"

    if cfg_obj["mixed_precision"] == "fp16":
        cmd += f" --mixed_precision=\"fp16\""
        if cfg_obj["full_16"]:
            cmd += f" --full_fp16"
    elif cfg_obj["mixed_precision"] == "bf16":
        cmd += f" --mixed_precision=\"bf16\""
        if cfg_obj["full_16"]:
            cmd += f" --full_bf16"

    cmd += " --gradient_checkpointing"
    cmd += " --learning_rate=\"{}\"".format(cfg_obj["learning_rate"])
    cmd += " --learning_rate_te1=\"{}\"".format(cfg_obj["learning_rate_te1"])
    cmd += " --learning_rate_te2=\"{}\"".format(cfg_obj["learning_rate_te2"])
    cmd += " --logging_dir=\"{}\"".format(cfg_obj["logging_dir"])
    cmd += " --lr_scheduler=\"{}\"".format("cosine")
    cmd += " --lr_scheduler_num_cycles=\"{}\"".format(1)
    cmd += " --max_data_loader_n_workers=\"{}\"".format(0)
    cmd += " --resolution=\"{}\"".format(cfg_obj["max_resolution"])
    cmd += " --max_train_epochs={}".format(cfg_obj["max_train_epochs"])
    cmd += " --min_snr_gamma={}".format(cfg_obj["min_snr_gamma"])
    if cfg_obj["noise_offset_type"] == "Multires":
        cmd += " --multires_noise_iterations=\"{}\"".format(cfg_obj["multires_noise_iterations"])
        cmd += " --multires_noise_discount=\"{}\"".format(cfg_obj["multires_noise_discount"])
    else:
        cmd += " --noise_offset=\"{}\"".format(cfg_obj["noise_offset"])
        cmd += " --adaptive_noise_scale=\"{}\"".format(cfg_obj["adaptive_noise_scale"])

    cmd += " --optimizer_type=\"{}\"".format(cfg_obj["optimizer"])
    cmd += " --output_dir=\"{}\"".format(cfg_obj["output_dir"])
    cmd += " --output_name=\"{}\"".format(cfg_obj["output_name"])
    cmd += " --pretrained_model_name_or_path=\"{}\"".format(cfg_obj["base_model"])
    cmd += " --save_model_as=\"{}\"".format("safetensors")
    cmd += " --save_precision=\"{}\"".format(cfg_obj["save_precision"])
    if cfg_obj["shuffle_caption"]:
        cmd += " --shuffle_caption"
    cmd += " --train_batch_size=\"{}\"".format(1)
    cmd += " --train_data_dir=\"{}\"".format(cfg_obj["train_data_dir"])
    cmd += " --vae=\"{}\"".format(cfg_obj["vae"])
    cmd += " --train_text_encoder"
    cmd += " --sample_at_first"
    cmd += " --loss_type \"l2\""

    if cfg_obj["xformers"]:
        cmd += " --{}".format(cfg_obj["xformers"])

    cmd += " --cus_lr_schedule=\"{}\"".format(cfg_obj["cus_lr_schedule"])

    make_prompt_file(cfg_obj["sample_tx_train_dir"])
    cmd += " --sample_prompts=\"{}\"".format(cfg_obj["sample_tx_train_dir"] + "\\sample\\prompt.json")
    cmd += " --sample_every_n_epochs=\"{}\"".format(cfg_obj["sample_every_n_epochs"])
    cmd += " --save_every_n_epochs=\"{}\"".format(cfg_obj["save_every_n_epochs"])
    cmd += " --sample_sampler=euler_a"

    if cfg_obj["ch_debug_dataset"]:
        cmd += f" --debug_dataset"

    return cmd


def save_train_info(cfg_obj, save_path, time_str):
    source_model = cfg_obj["base_model"]
    info_obj = {}
    if source_model and os.path.exists(source_model):
        info_obj["base_model"] = os.path.basename(source_model)
        base_mode_time = os.path.getmtime(source_model)
        modified_time = datetime.datetime.fromtimestamp(base_mode_time)
        formatted_time = modified_time.strftime('%Y-%m-%d %H:%M:%S')
        info_obj["base_model_date"] = formatted_time
    save_file = os.path.join(save_path, "TrainInfo_" + time_str + ".json")
    save_prompt_info(cfg_obj["train_data_dir"], save_path)
    with open(save_file, 'w') as f:
        json.dump(info_obj, f, indent=2)


def save_prompt_info(dataset_path, save_path):
    import time
    timestamp = time.time()
    time_tuple = time.localtime(timestamp)
    formatted_time = time.strftime("%m-%d-%H-%M-%S", time_tuple)
    if dataset_path and os.path.exists(dataset_path):
        info_obj = {}
        prompt_stat = {}
        img_count = 0
        for r, d, fs in os.walk(dataset_path):
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
        info_obj["data_type"] = "PromptInfo"
        info_obj["img_count"] = img_count
        info_obj["p_stat"] = p_stat_list
    save_file = os.path.join(save_path, "PromptInfo_" + formatted_time + ".json")
    with open(save_file, 'w') as f:
        json.dump(info_obj, f, indent=2)


def make_flux_lora_cmd(cfg_obj):
    cmd = "accelerate launch"
    cmd += f" --num_cpu_threads_per_process={2}"

    cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "flux_train_network.py") + "\""


    cmd += f" --bucket_no_upscale"
    cmd += f" --network_module=\"networks.lora_flux\""
    cmd += f" --bucket_reso_steps={256}"
    cmd += f" --cache_latents"
    cmd += f" --caption_extension=\".txt\""
    cmd += f" --enable_bucket"
    cmd += f" --min_bucket_reso={768}"
    cmd += f" --max_bucket_reso={1024}"

    if cfg_obj["mixed_precision"] == "fp16":
        cmd += f" --mixed_precision=\"fp16\""
        if cfg_obj["full_16"]:
            cmd += f" --full_fp16"
    elif cfg_obj["mixed_precision"] == "bf16":
        cmd += f" --mixed_precision=\"bf16\""
        if cfg_obj["full_16"]:
            cmd += f" --full_bf16"
    cmd += " --network_dim={}".format(cfg_obj["network_dim"])
    cmd += " --network_alpha={}".format(cfg_obj["network_alpha"])
    cmd += " --gradient_checkpointing"
    if cfg_obj["optimizer"] == "Prodigy":
        cmd += " --learning_rate=\"{}\"".format(1)
        cmd += " --text_encoder_lr=\"{}\"".format(1)
        cmd += " --optimizer_args {}".format("weight_decay=0.01 betas=.9,.99 decouple=True use_bias_correction=True d_coef=0.5 d0=1e-5")
    else:
        cmd += " --learning_rate=\"{}\"".format(cfg_obj["learning_rate"])
        cmd += " --unet_lr=\"{}\"".format(cfg_obj["learning_rate"])
        cmd += " --text_encoder_lr=\"{}\"".format(cfg_obj["text_encoder_lr"])
    cmd += " --logging_dir=\"{}\"".format(cfg_obj["logging_dir"])
    cmd += " --lr_scheduler=\"{}\"".format("constant")
    cmd += " --lr_scheduler_num_cycles=\"{}\"".format(1)
    cmd += " --max_data_loader_n_workers=\"{}\"".format(0)
    cmd += " --resolution=\"{}\"".format(cfg_obj["max_resolution"])
    cmd += " --max_train_epochs={}".format(cfg_obj["max_train_epochs"])
    cmd += " --min_snr_gamma={}".format(cfg_obj["min_snr_gamma"])
    if cfg_obj["noise_offset_type"] == "Multires":
        cmd += " --multires_noise_iterations=\"{}\"".format(cfg_obj["multires_noise_iterations"])
        cmd += " --multires_noise_discount=\"{}\"".format(cfg_obj["multires_noise_discount"])
    else:
        cmd += " --noise_offset=\"{}\"".format(cfg_obj["noise_offset"])
        cmd += " --adaptive_noise_scale=\"{}\"".format(cfg_obj["adaptive_noise_scale"])

    cmd += " --optimizer_type=\"{}\"".format(cfg_obj["optimizer"])
    cmd += " --output_dir=\"{}\"".format(cfg_obj["output_dir"])
    cmd += " --output_name=\"{}\"".format(cfg_obj["output_name"])
    cmd += " --pretrained_model_name_or_path=\"{}\"".format(cfg_obj["base_model"])
    if cfg_obj["network_weights"]:
        cmd += " --network_weights=\"{}\"".format(cfg_obj["network_weights"])
    cmd += " --save_model_as=\"{}\"".format("safetensors")
    cmd += " --save_precision=\"{}\"".format(cfg_obj["save_precision"])
    if cfg_obj["shuffle_caption"]:
        cmd += " --shuffle_caption"
    cmd += " --train_batch_size=\"{}\"".format(cfg_obj["train_batch_size"])
    cmd += " --train_data_dir=\"{}\"".format(cfg_obj["train_data_dir"])
    cmd += " --ae=\"{}\"".format(cfg_obj["vae"])
    cmd += " --sample_at_first"
    cmd += " --loss_type \"l2\""

    if cfg_obj["xformers"]:
        cmd += " --{}".format(cfg_obj["xformers"])

    cmd += " --cus_lr_schedule=\"{}\"".format(cfg_obj["cus_lr_schedule"])

    make_prompt_file(cfg_obj["sample_tx_train_dir"])
    cmd += " --sample_prompts=\"{}\"".format(cfg_obj["sample_tx_train_dir"] + "\\sample\\prompt.json")
    cmd += " --sample_every_n_epochs=\"{}\"".format(cfg_obj["sample_every_n_epochs"])
    cmd += " --save_every_n_epochs=\"{}\"".format(cfg_obj["save_every_n_epochs"])
    cmd += " --sample_sampler=euler_a"

    if cfg_obj["ch_debug_dataset"]:
        cmd += f" --debug_dataset"

    cmd += " --clip_l=./clip/clip_l.safetensors"
    cmd += " --t5xxl=./clip/t5xxl_fp16.safetensors"
    cmd += " --apply_t5_attn_mask"
    cmd += " --model_prediction_type=raw --guidance_scale=1 --guidance_rescale"
    cmd += " --network_train_unet_only"
    cmd += " --fp8_base"


    return cmd


def make_lora_cmd(cfg_obj):
    cmd = "accelerate launch"
    cmd += f" --num_cpu_threads_per_process={2}"
    if cfg_obj["sdxl"]:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "sdxl_train_network.py") + "\""
    else:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "train_network.py") + "\""

    cmd += f" --bucket_no_upscale"
    cmd += f" --network_module=\"networks.lora\""
    cmd += f" --bucket_reso_steps={64}"
    cmd += f" --cache_latents"
    cmd += f" --caption_extension=\".txt\""
    cmd += f" --enable_bucket"
    cmd += f" --min_bucket_reso={640}"
    cmd += f" --max_bucket_reso={1536}"

    if cfg_obj["mixed_precision"] == "fp16":
        cmd += f" --mixed_precision=\"fp16\""
        if cfg_obj["full_16"]:
            cmd += f" --full_fp16"
    elif cfg_obj["mixed_precision"] == "bf16":
        cmd += f" --mixed_precision=\"bf16\""
        if cfg_obj["full_16"]:
            cmd += f" --full_bf16"
    cmd += " --network_dim={}".format(cfg_obj["network_dim"])
    cmd += " --network_alpha={}".format(cfg_obj["network_alpha"])
    cmd += " --gradient_checkpointing"
    if cfg_obj["optimizer"] == "Prodigy":
        cmd += " --learning_rate=\"{}\"".format(1)
        cmd += " --text_encoder_lr=\"{}\"".format(1)
        cmd += f" --optimizer_args weight_decay=0.01 betas=.9,.99 decouple=True use_bias_correction=True d_coef=0.5 d0={cfg_obj["d0"]}"
    else:
        cmd += " --learning_rate=\"{}\"".format(cfg_obj["learning_rate"])
        cmd += " --text_encoder_lr=\"{}\"".format(cfg_obj["text_encoder_lr"])
    cmd += " --logging_dir=\"{}\"".format(cfg_obj["logging_dir"])
    cmd += " --lr_scheduler=\"{}\"".format("constant")
    cmd += " --lr_scheduler_num_cycles=\"{}\"".format(1)
    cmd += " --max_data_loader_n_workers=\"{}\"".format(0)
    cmd += " --resolution=\"{}\"".format(cfg_obj["max_resolution"])
    cmd += " --max_train_epochs={}".format(cfg_obj["max_train_epochs"])
    cmd += " --min_snr_gamma={}".format(cfg_obj["min_snr_gamma"])
    if cfg_obj["noise_offset_type"] == "Multires":
        cmd += " --multires_noise_iterations=\"{}\"".format(cfg_obj["multires_noise_iterations"])
        cmd += " --multires_noise_discount=\"{}\"".format(cfg_obj["multires_noise_discount"])
    else:
        cmd += " --noise_offset=\"{}\"".format(cfg_obj["noise_offset"])
        cmd += " --adaptive_noise_scale=\"{}\"".format(cfg_obj["adaptive_noise_scale"])

    cmd += " --optimizer_type=\"{}\"".format(cfg_obj["optimizer"])
    cmd += " --output_dir=\"{}\"".format(cfg_obj["output_dir"])
    cmd += " --output_name=\"{}\"".format(cfg_obj["output_name"])
    cmd += " --pretrained_model_name_or_path=\"{}\"".format(cfg_obj["base_model"])
    if cfg_obj["network_weights"]:
        cmd += " --network_weights=\"{}\"".format(cfg_obj["network_weights"])
    cmd += " --save_model_as=\"{}\"".format("safetensors")
    cmd += " --save_precision=\"{}\"".format(cfg_obj["save_precision"])
    if cfg_obj["shuffle_caption"]:
        cmd += " --shuffle_caption"
    cmd += " --train_batch_size=\"{}\"".format(cfg_obj["train_batch_size"])
    cmd += " --train_data_dir=\"{}\"".format(cfg_obj["train_data_dir"])
    if cfg_obj["vae"]:
        cmd += " --vae=\"{}\"".format(cfg_obj["vae"])
    cmd += " --sample_at_first"
    cmd += " --loss_type \"l2\""

    if cfg_obj["xformers"]:
        cmd += " --{}".format(cfg_obj["xformers"])

    cmd += " --cus_lr_schedule=\"{}\"".format(cfg_obj["cus_lr_schedule"])

    make_prompt_file(cfg_obj["sample_tx_train_dir"])
    cmd += " --sample_prompts=\"{}\"".format(cfg_obj["sample_tx_train_dir"] + "\\sample\\prompt.json")
    cmd += " --sample_every_n_epochs=\"{}\"".format(cfg_obj["sample_every_n_epochs"])
    cmd += " --save_every_n_epochs=\"{}\"".format(cfg_obj["save_every_n_epochs"])
    cmd += " --sample_sampler=euler_a"

    if cfg_obj["ch_debug_dataset"]:
        cmd += f" --debug_dataset"

    cmd += f" --caption_tag_dropout_rate={0.5}"
    return cmd



def make_extract_lora_from_model_cmd(cfg_obj):
    cmd = os.getcwd() + "/venv/Scripts/python.exe "
    cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "networks", "extract_lora_from_models.py") + "\""
    cmd += f" --model_org=\"{cfg_obj['model_org']}\""
    cmd += f" --model_tuned=\"{cfg_obj['model_tuned']}\""
    cmd += f" --dim={cfg_obj['dim']}"
    if cfg_obj["save_precision"] != "None":
        cmd += f" --save_precision=\"{cfg_obj['save_precision']}\""
    if cfg_obj["load_precision"] != "None":
        cmd += f" --load_precision=\"{cfg_obj['load_precision']}\""
    cmd += f" --device=\"{cfg_obj['device']}\""
    cmd += f" --save_to=\"{cfg_obj['save_to']}\""
    if cfg_obj["model_type"] == "SDXL":
        cmd += " --sdxl"
    elif cfg_obj["model_type"] == "V2":
        cmd += " --v2"

    return cmd
