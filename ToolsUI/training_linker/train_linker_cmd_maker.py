"""
FILE_NAME: data_set_cmd_maker.py
AUTHOR: RCB
CREATED: 2024/4/2-11:54
DESC: 
"""
import json
import os
from PIL import Image


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


def make_lora_cmd(cfg_obj):
    cmd = "accelerate launch"
    cmd += f" --num_cpu_threads_per_process={2}"
    if cfg_obj["sdxl"]:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "sdxl_train_network.py") + "\""
    else:
        cmd += " \"" + os.path.join(os.getcwd(), "sd-script", "train_network.py") + "\""

    cmd += f" --network_module=\"networks.lora\""

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
    cmd += " --network_dim={}".format(cfg_obj["network_dim"])
    cmd += " --network_alpha={}".format(cfg_obj["network_alpha"])
    cmd += " --gradient_checkpointing"
    cmd += " --learning_rate=\"{}\"".format(cfg_obj["learning_rate"])
    cmd += " --text_encoder_lr=\"{}\"".format(cfg_obj["text_encoder_lr"])
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
    cmd += " --train_batch_size=\"{}\"".format(cfg_obj["train_batch_size"])
    cmd += " --train_data_dir=\"{}\"".format(cfg_obj["train_data_dir"])
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

    return cmd
