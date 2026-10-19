import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from tqdm import tqdm

##### YOUR CONFIG #####

# JSON format: {site: {subject: absolute NIfTI path}}.
jsonfile = ""
target_path = ""

######################

def _worker_one_subject(args):
    jsonfile, savepath, cut_direct, key_site, key_subject, subject_path, site_id = args
    reader = datareader(jsonfile, savepath, cut_direct)

    img_1c_outpack, text_outpack, subject_max, subject_min = reader.slicing(
        subject_path, cut_direct
    )
    one_subject_info = reader.saving_slices(
        img_1c_outpack, text_outpack, key_site, key_subject, site_id
    )

    return key_site, key_subject, one_subject_info, subject_max, subject_min


class datareader():
    def __init__(self, jsonfile, savepath, cut_direct):
        self.jsonfile = jsonfile
        self.savepath = savepath
        self.cut_direct = cut_direct
        self.Channel_num = "OneChannel"

    def saving_slices(self, img_packin, text_packin, x_site, x_subject, id_site):
        datasavepath = os.path.join(
            self.savepath, self.Channel_num, "cut_direct{}".format(self.cut_direct),
            "data_storage", x_site, x_subject
        )
        os.makedirs(datasavepath)

        subject_slice_npy_pack = np.stack(img_packin, axis=0)
        subject_savepath = os.path.join(
            datasavepath,
            text_packin[0][0] + "_" + text_packin[0][1] + "_cutd" + text_packin[0][3] + ".npy"
        )
        np.save(subject_savepath, subject_slice_npy_pack)

        data_list = []
        for info in text_packin:
            one_info = dict()
            one_info["site"] = info[0]
            one_info["subject"] = info[1]
            one_info["#slice"] = info[2]
            one_info["cut_direction"] = info[3]
            one_info["img"] = subject_savepath

            one_info["img_aug"] = subject_savepath
            one_info["#Channel"] = self.Channel_num
            one_info["site_id"] = id_site
            data_list.append(one_info)

        return data_list

    def slicing(self, img_inpath, cut_direction=3):
        img = nib.load(img_inpath).get_fdata(dtype=np.float32)

        image_data_list_one_channel = []
        text_data_list = []
        max_list = []
        min_list = []

        axis_len = img.shape[cut_direction - 1]
        for cut in range(axis_len):
            oneC_data_cut = np.take(img, indices=cut, axis=cut_direction - 1)
            max_list.append(np.max(oneC_data_cut))
            min_list.append(np.min(oneC_data_cut))
            image_data_list_one_channel.append(oneC_data_cut)

            site_name = img_inpath.split("/")[-2]
            subject_name = img_inpath.split("/")[-1]
            one_text_data_list = []
            one_text_data_list.append(site_name)
            one_text_data_list.append(subject_name)
            text = str(cut).zfill(3)
            one_text_data_list.append(text)
            one_text_data_list.append(str(cut_direction))
            text_data_list.append(one_text_data_list)

        max_value = np.max(np.array(max_list))
        min_value = np.min(np.array(min_list))

        return image_data_list_one_channel, text_data_list, max_value, min_value

    def process_parallel(self, num_workers=8):
        with open(self.jsonfile, "r", encoding="utf-8") as f:
            loaded_datalist = json.load(f)

        global_min = float("inf")
        global_max = float("-inf")
        print("Processing cut direction:", self.cut_direct)

        tasks = []

        site_order = list(loaded_datalist.keys())
        site_to_subject_infos = {s: {} for s in site_order}

        for site_id, key_site in enumerate(site_order):
            for key_subject in loaded_datalist[key_site]:
                subject_path = loaded_datalist[key_site][key_subject]
                tasks.append((
                    self.jsonfile, self.savepath, self.cut_direct,
                    key_site, key_subject, subject_path, site_id
                ))

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_worker_one_subject, t) for t in tasks]
            for fu in tqdm(as_completed(futures), total=len(futures)):
                result = fu.result()
                key_site, key_subject, one_subject_info, subject_max, subject_min = result

                global_min = min(global_min, float(subject_min))
                global_max = max(global_max, float(subject_max))
                site_to_subject_infos[key_site][key_subject] = one_subject_info

        out_datalist = []
        for key_site in site_order:
            site_datalist = []
            for key_subject in loaded_datalist[key_site]:
                one_subject_info = site_to_subject_infos[key_site][key_subject]
                site_datalist.extend(one_subject_info)
            out_datalist.append(site_datalist)

        json_path = os.path.join(
            self.savepath, self.Channel_num, "cut_direct{}".format(self.cut_direct)
        )
        os.makedirs(json_path, exist_ok=True)

        out_maxmin = {"max": float(global_max), "min": float(global_min)}
        output_maxmin = os.path.join(json_path, "max_min_value.json")
        with open(output_maxmin, "w", encoding="utf-8") as f:
            json.dump(out_maxmin, f, indent=4, ensure_ascii=False)

        print("max_graylevel:", global_max)
        print("min_graylevel:", global_min)

        output_json = os.path.join(json_path, "datalist.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out_datalist, f, indent=4, ensure_ascii=False)


# cut_direct is the array axis (1, 2, or 3); adjust num_workers for your machine.
reader = datareader(jsonfile, target_path, cut_direct=3)
reader.process_parallel(num_workers=30)