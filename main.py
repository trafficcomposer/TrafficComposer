import os
import time

from gen_traffic_ir.gen_multi_modality import MultiModal


if __name__ == '__main__':
    runner = MultiModal()
    
    dir_text_file:str 
    
    dir_lane_detection_file:str
    dir_vehicle_detection_file:str

    dir_img_file:str

    dir_text_ir:str
    dir_img_ir:str
    dir_merged_ir:str

    ls_text_file = []
    for fn in os.listdir(dir_text_file):
        if fn.endswith('.txt'):
            path_text_file = os.path.join(dir_text_file, fn)
            ls_text_file.append(path_text_file)
    ls_text_file.sort()

    counter_exist, counter_missing = 0, 0
    for path_text_file in ls_text_file:
        text_fn = path_text_file.split('/')[-1]
        str_file_idx = text_fn.split('_')[-1].split('.')[0]

        fn_lane_detection = f"{str_file_idx}.lines.txt"

        fpath_vehicle_detection = os.path.join(dir_vehicle_detection_file, text_fn)
        fpath_lane_detection = os.path.join(dir_lane_detection_file, fn_lane_detection)

        fpath_text_ir = os.path.join(dir_text_ir, text_fn)
        fpath_img_ir = os.path.join(dir_img_ir, f"{str_file_idx}.yaml")
        fpath_merged_ir = os.path.join(dir_merged_ir, f"{str_file_idx}.yaml")

        # ==================================================
        # ##         Parse the text file with GPT         ##
        # ==================================================
        if not os.path.exists(fpath_text_ir):
            print(f"Run GPT to generate text_ir for {text_fn}")
            input("Press Enter to continue...")
            runner.gen_text_ir(path_text_file, fpath_text_ir)
            time.sleep(10)
        # else:
        #     print(f"No need to run GPT, {fpath_text_ir} exists.")

        # ==================================================
        # ##         Parse the image file with IR         ##
        # ==================================================
        if not os.path.exists(fpath_img_ir):
            try:
                assert os.path.exists(fpath_lane_detection)
                assert os.path.exists(fpath_vehicle_detection)
                print(f"Generating img_ir for {str_file_idx}.yaml")
                runner.gen_img_ir(fpath_lane_detection, fpath_vehicle_detection, fpath_img_ir)
            except:
                print(f"Missing {fpath_lane_detection} or {fpath_vehicle_detection}")
        # else:
        #     print(f"No need to generate Image IR, {fpath_img_ir} exists.")
        
        # ==================================================
        # ##         Merge two modalities into one        ##
        # ==================================================
        if not os.path.exists(fpath_merged_ir):
            try:
                assert os.path.exists(fpath_text_ir)
                assert os.path.exists(fpath_img_ir)
                print(f"Merge two modalities into one for {str_file_idx}.yaml")
                runner.merge_two_modality(fpath_text_ir, fpath_img_ir, fpath_merged_ir)
            except:
                print(f"Missing {fpath_text_ir} or {fpath_img_ir}")
        # else:
        #     print(f"No need to merge two modalities, {fpath_merged_ir} exists.")
