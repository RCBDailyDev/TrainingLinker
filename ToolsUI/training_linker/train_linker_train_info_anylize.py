import os


def is_number(s):
    return s.isdigit()


def is_dataset(dir_name):
    sl = dir_name.split("_")
    if len(sl) > 1:
        if is_number(sl[0]):
            return True
    return False


def get_train_info(cfg_object):
    train_data_dir = cfg_object["train_data_dir"]
    dir_list = []
    for root, dirs, files in os.walk(train_data_dir):
        for d in dirs:
            if is_dataset(d):
                dir_list.append((d, os.path.join(root, d)))

    total_img = 0
    step_per_epoch = 0

    for (d, p) in dir_list:
        for root, dirs, files in os.walk(p):
            repeat = int(d.split("_")[0])
            for f in files:
                if f.endswith((".png", ".jpg")):
                    total_img += 1
                    step_per_epoch += repeat

    ##TODO:DELETE
    print(f"Total Image: {total_img}")
    print(f"Steps: {step_per_epoch}")
    print("Total Steps: {}".format(step_per_epoch * cfg_object["max_train_epochs"]))
