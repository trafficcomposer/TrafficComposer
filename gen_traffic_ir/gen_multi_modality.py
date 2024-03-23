import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import json
import yaml
import copy

from pprint import pprint


from ultralytics import YOLO


from gen_merged_ir.gpt_text_parser import parse_scenario_all_in_one


class MultiModal(object):
    """
    To run the YOLOv8 model
    """
    def __init__(self, data_path=None):
        self.data_path = data_path

        self.detect_path = "src/yolov8/runs/detect/predict2"
        self.transform_matrix_path = "src/yolov8/transforms.json"
        self.is_debug = False

        with open(self.transform_matrix_path, "r") as f:
            self.ls_transform_matrix = json.load(f)['frames']
        
        with open('clrnet/coco.yaml', 'r') as f:
            self.dct_coco = yaml.load(f, Loader=yaml.FullLoader)['names']
        # print(self.dct_coco)

        self.out_path = "out/"
    
    def yolo_detect(self):
        """Detect
        Results will be saved to runs/detect
        """
        # Load a model
        model = YOLO("yolov8n.yaml").load('yolov8n.pt')

        # Use the model
        # model.train(data="coco128.yaml", epochs=3)  # train the model
        # metrics = model.val()  # evaluate model performance on the validation set
        # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        results = model(self.data_path, save=True, save_txt=True)  # predict on an image
        # path = model.export(format="onnx")  # export the model to ONNX format

    def yolo_segment(self):
        model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
        results = model(self.data_path, save=True, save_txt=True)  # predict on an image

    def debug(self):
        self.isdebug = True
    
    def visualize(self, img, coord_xyxy, coord_xywh):
        # Note that the coordinates are in (x, y, x_len, y_len) format
        # (x, y) is the upper left corner of the rectangle
        cv2.rectangle(img, (coord_xyxy[0], coord_xyxy[1], coord_xywh[2], coord_xywh[3]), (0, 255, 0), 2)
        cv2.circle(img, (coord_xyxy[0], coord_xyxy[1]), 50, (0, 0, 255), -1)
        cv2.circle(img, (coord_xyxy[2], coord_xyxy[3]), 50, (0, 0, 255), -1)
        cv2.imwrite(f"tmp.png", img)

    def assign_vehicle_lane_single(self, coord_left_bottom, coord_right_bottom, ls_lines):
        threshold = 0.6
        ls_lines_key = []
        for idx, line in enumerate(ls_lines):
            is_found = False
            for x, y in line:
                if coord_left_bottom[1] > y:
                    ls_lines_key.append((x, y))
                    is_found = True
                    break
            
            if idx == 0 and not is_found:
                # The vehicle is located on the left side of the lane
                ls_lines_key.append(coord_left_bottom)
            elif idx == len(ls_lines)-1 and not is_found:
                # The vehicle is located on the right side of the lane
                ls_lines_key.append(coord_right_bottom)
        
        if not (len(ls_lines_key) == len(ls_lines)):
            return None

        for i in range(len(ls_lines_key)-2):
            x, y = ls_lines_key[i]
            if coord_left_bottom[0] <= x:
                if coord_right_bottom[0] <= x:
                    # The vehicle is located on the left side of the lane
                    return i-1
                else:
                    # coord_right_bottom[0] > x
                    if coord_right_bottom[0] <= ls_lines_key[i+1][0]:
                        # The vehicle is stepping on the line i, to decide which lane it belongs to
                        if abs(coord_left_bottom[0]-x) / (abs(coord_left_bottom[0]-x) + abs(coord_right_bottom[0]-x)) > threshold:
                            return i-1
                        else:
                            return i
                    else:
                        # coord_right_bottom[0] >= ls_lines_key[i+1][0]
                        # The vehicle is wider than the lane, it belongs to the line i
                        return i
            else:
                # coord_left_bottom[0] > x, continue
                continue
        
        i = len(ls_lines_key) - 1
        
        x, y = ls_lines_key[i]

        if coord_left_bottom[0] <= x:
            if coord_right_bottom[0] <= x:
                # The vehicle is located on the left side of the lane
                return i-1
            else:
                # coord_right_bottom[0] > x
                
                # The vehicle is stepping on the line i, to decide which lane it belongs to
                if abs(coord_left_bottom[0]-x) / (abs(coord_left_bottom[0]-x) + abs(coord_right_bottom[0]-x)) > threshold:
                    return i-1
                else:
                    return i
        else:
            # coord_left_bottom[0] > x, continue
            return i

    def assign_vehicle_lane(self, lane_detection_fp=None, vehicle_detection_fp=None, save_path=None, img_w=1640, img_h=590):
        assert os.path.exists(lane_detection_fp)
        assert os.path.exists(vehicle_detection_fp)

        # ## load the lane detection results
        with open(lane_detection_fp, "r") as f:
            ls_lines_raw = f.readlines()

        # ## For visualization load the image
        if self.is_debug:
            # ## for visualization
            # ls_split = lane_detection_path.split("/")
            # img_name = f"{ls_split[-2]}_{ls_split[-1]}_{file_idx:05d}.jpg"
            # img_path = os.path.join(self.data_path, img_name)
            # img = cv2.imread(img_path)
            pass
        
        # ## extract all the lines
        ls_lines = []
        for line in ls_lines_raw:
            line = line.strip("\n")
            ls_line = line.split(" ")

            ls_line = [int(float(x)) for x in ls_line]
            ls_line_coor = list(zip(ls_line[0::2], ls_line[1::2]))
            ls_lines.append(ls_line_coor)

        # ## rank the lines from left to right
        ls_lines.sort(key=lambda x: x[0][0])

        # ## Check wheter we can detect lanes
        if len(ls_lines) == 0:
            return None

        # ## load the vehicle detection results
        with open(vehicle_detection_fp, "r") as f:
            ls_vehicles_raw = f.readlines()
        ls_vehicles_raw = [x.strip("\n") for x in ls_vehicles_raw]

        # ## assign lane for each vehicle
        dct_lane_vehicle = {}
        for vehicle in ls_vehicles_raw:
            ls_vehicle = vehicle.split(" ")

            xywhn = (float(ls_vehicle[1]), float(ls_vehicle[2]), float(ls_vehicle[3]), float(ls_vehicle[4]))
            xyxyn = (
                xywhn[0]-xywhn[2]/2, 
                xywhn[1]-xywhn[3]/2, 
                xywhn[0]+xywhn[2]/2, 
                xywhn[1]+xywhn[3]/2
            )
            coord_xyxy = (
                int(xyxyn[0]*img_w),
                int(xyxyn[1]*img_h),
                int(xyxyn[2]*img_w),
                int(xyxyn[3]*img_h)
            )
            coord_left_bottom = (coord_xyxy[0], coord_xyxy[3])
            coord_right_bottom = (coord_xyxy[2], coord_xyxy[3])

            lane_idx = self.assign_vehicle_lane_single(coord_left_bottom, coord_right_bottom, ls_lines)
            if lane_idx is None:
                print("Cannot find the lane for the vehicle")
                if self.is_debug:
                    # # ## for visualization
                    # ls_split = lane_detection_path.split("/")
                    # img_name = f"{ls_split[-2]}_{ls_split[-1]}_{file_idx:05d}.jpg"
                    # img_path = os.path.join(self.data_path, img_name)

                    # img = cv2.imread(img_path)

                    # cv2.circle(img, coord_left_bottom, 50, (0, 255, 0), -1)
                    # cv2.circle(img, coord_right_bottom, 50, (0, 255, 0), -1)
                    # cv2.imwrite("tmp.png", img)
                    input("Press Enter to continue...")
            else:
                if lane_idx not in dct_lane_vehicle:
                    dct_lane_vehicle[lane_idx] = [vehicle]
                else:
                    dct_lane_vehicle[lane_idx].append(vehicle)

        # ## assign lane for the ego vehicle
        x_middle = img_w // 2
        y_bottom = img_h

        # ## Assign lane to ego vehicle
        # ## Select the lane the vertical line is in
        ego_line_idx = self.assign_vehicle_lane_single((x_middle-1, y_bottom), (x_middle+1, y_bottom), ls_lines)

        # # ## Select the lane with the maximum horizontal line
        # left_line = ls_lines[0]
        # left_line_y_max = max([y for x, y in left_line])
        # right_lines = ls_lines[-1]
        # right_line_y_max = max([y for x, y in right_lines])
        # line_y_max = max(left_line_y_max, right_line_y_max)

            
        # ## Post-processing
        # ## assign vehicle name to each vehicle
        # print(dct_lane_vehicle)
        dct_lane_vehicle_name = {}
        for lane in list(dct_lane_vehicle.keys()):
            dct_lane_vehicle_name[lane] = []
            for vehicle in dct_lane_vehicle[lane]:
                vehicle_name = self.dct_coco[int(vehicle.split(" ")[0])]
                dct_lane_vehicle_name[lane].append((vehicle_name, vehicle))
        
        if ego_line_idx in dct_lane_vehicle_name:
            dct_lane_vehicle_name[ego_line_idx].append(("ego", "ego 0 0 0 0"))
        else:
            dct_lane_vehicle_name[ego_line_idx] = [("ego", "ego 0 0 0 0")]
        

        # print(dct_lane_vehicle_name)
        
        if self.is_debug:
            # ## for visualization

            # ## load image
            # ls_split = lane_detection_path.split("/")
            # img_name = f"{ls_split[-2]}_{ls_split[-1]}_{file_idx:05d}.jpg"
            # img_path = os.path.join(self.data_path, img_name)
            # img = cv2.imread(img_path)

            # line = ls_lines[0]
            # # # for line in ls_lines:
            # # for x, y in line:
            # #     cv2.circle(img, (x, y), 50, (0, 0, 255), -1)
            # # # break
            # x, y = line[0]
            # print(x, y)
            # cv2.circle(img, (x, y), 50, (0, 255, 0), -1)
            # # cv2.circle(img, (x+60, y), 50, (0, 255, 0), -1)
            # cv2.circle(img, (x, y+60), 50, (0, 0, 255), -1)
            # # cv2.circle(img, (x, y-60), 50, (0, 0, 255), -1)
            # # for y in range(0, img_h, 100):
            # #     cv2.circle(img, (x, y), 50, (0, 0, 255), -1)

            # cv2.imwrite("tmp.png", img)
            pass

        with open(save_path, "w") as f:
            yaml.dump(dct_lane_vehicle_name, f)

        return dct_lane_vehicle_name

    def merge_two_modality(self, yml_text, yml_img, save_fn=None):
        """
        Merge the text and image modalities
        """
        dct_direction = {
            "left": "left",
            "right": "right",
            "ahead": "ahead",
            "in front": "ahead",
            "behind": "behind",
        }

        # ## load the text modality
        if type(yml_text) == str:
            with open(yml_text, "r") as f:
                txt = f.read()
                dct_text = yaml.safe_load(txt)
            if type(dct_text) == str:
                dct_text = yaml.load(dct_text, Loader=yaml.FullLoader)
        else:
            dct_text = copy.copy(yml_text)

        # ## load the image modality
        if type(yml_img) == str:
            with open(yml_img, "r") as f:
                dct_img = yaml.load(f, Loader=yaml.FullLoader)
        else:
            dct_img = copy.copy(yml_img)

        if dct_text == None:
            dct_ans = dct_img
        elif dct_img == None:
            dct_ans = dct_text
        if dct_text == None or dct_img == None:
            # ## save the merged modality
            if save_fn is None:
                save_fn = "merged.yaml"
            with open(save_fn, "w") as f:
                yaml.dump(dct_ans, f)
            return dct_ans
        
        if self.is_debug:
            print("dct_text:")
            pprint(dct_text)
            print()
            print(f"dct_img:")
            pprint(dct_img)
            print()
            print(f"type(dct_text): {type(dct_text)}")
            print(f"type(dct_img): {type(dct_img)}")

        # ## register all the detected vehicles in dct_img
        dct_vehicle = {}
        for lane in dct_img.keys():
            dct_vehicle[lane] = []
            for vehicle in dct_img[lane]:
                dct_vehicle[lane].append(False)
        
        # for idx_lane in sorted(list(dct_img.keys())):

        dct_ans = copy.copy(dct_text)

        if self.is_debug:
            print(f"dct_ans: {dct_ans}")
            print(f"type(dct_ans): {type(dct_ans)}")
            print()
        dct_ans['lane_num'] = len(list(dct_img.keys()))

        # ## look for the ego vehicle
        # ## To find the lane index of the ego vehicle
        for idx_lane in sorted(list(dct_img.keys())):
            # print(f"idx_lane: {idx_lane}")
            for idx_vehicle, vehicle in enumerate(dct_img[idx_lane]):
                if "ego" == vehicle[0]:
                    dct_ans['participant']['ego_vehicle']['lane_idx'] = idx_lane
                    ego_lane_idx = idx_lane

                    dct_vehicle[idx_lane][idx_vehicle] = True
                    break

        for pariticipant in dct_text['participant']:

            if self.is_debug:
                print(f"participant: {pariticipant}")

            if pariticipant == "ego_vehicle":
                continue
            else:
                # ## look for the other vehicles
                # ## To find the lane index of the ego vehicle
                # ## According to the vehicle position
                # ## Assumption: the vehicle is the nearest one to the ego vehicle in the direction
                direction = None
                for pos_direction in dct_text['participant'][pariticipant]['position_target']:
                    if pos_direction in dct_direction.keys():
                        direction = dct_direction[pos_direction]
                        break
                if self.is_debug:
                    if direction is None:
                        print("Cannot find the direction")
                        input("Press Enter to continue...")
                
                if direction == "left":
                    lane_idx = ego_lane_idx - 1
                elif direction == "right":
                    lane_idx = ego_lane_idx + 1
                elif direction == "ahead" or "behind":
                    lane_idx = ego_lane_idx
                dct_ans['participant'][pariticipant]['lane_idx'] = lane_idx

                # ## Find the nearest vehicle in lane_idx
                if lane_idx != ego_lane_idx:
                    dct_vehicle[lane_idx][-1] = True
                else:
                    try:
                        # ## The vehicle is in the same lane as the ego vehicle
                        dct_vehicle[lane_idx][-2] = True
                    except:
                        pass
        idx_other_vehicle = len(dct_text['participant']) - 1

        if self.is_debug:
            print("dct_vehicle:")
            pprint(dct_vehicle)
        
        for lane in dct_vehicle.keys():
            for idx, vehicle in enumerate(dct_vehicle[lane]):
                if not vehicle:
                    # ## The vehicle is not registered in the text modality
                    # ## Register the vehicle in the text modality
                    if lane < ego_lane_idx:
                        direction = "left"
                    elif lane > ego_lane_idx:
                        direction = "right"
                    else:
                        direction = "ahead"
                    idx_other_vehicle += 1

                    dct_text['participant'][f"other_vehicle_{idx_other_vehicle}"] = {
                        "lane_idx": lane,
                        "position_target": [f"{direction}", "ego_vehicle"],
                        # "speed_target": 0,
                        # "speed_limit": 0,
                        # "speed_limit_unit": "km/h"
                        "type": dct_img[lane][idx][0],
                    }
                    dct_vehicle[lane][idx] = True
        if self.is_debug:
            print("dct_ans:")
            pprint(dct_ans)

        # ## save the merged modality
        if save_fn is None:
            save_fn = "merged.yaml"
        with open(save_fn, "w") as f:
            yaml.dump(dct_ans, f)
    
        return dct_ans

    def gen_text_ir(self, path_text_file, save_path):
        """
        Process the text file
        """
        # assert save_path.endswith(".json")
        assert not os.path.exists(save_path)
        assert os.path.exists(path_text_file)

        with open(path_text_file, 'r') as scenario_file:
            scenario = scenario_file.readlines()

        scenario = ''.join(scenario)
        assert scenario != ''
        
        scenario_info = parse_scenario_all_in_one(scenario)

        with open(save_path, 'w') as fp:
            json.dump(scenario_info, fp)

    def gen_img_ir(self, path_lane_detection=None, path_vehicle_detection=None, save_path=None, img_w=1640, img_h=590):
        return self.assign_vehicle_lane(path_lane_detection, path_vehicle_detection, save_path, img_w, img_h)

if __name__ == '__main__':
    runner = MultiModal(data_path: str)
    
    runner.merge_two_modality("out/case_txt.yaml", "out/case_img.yaml", save_fn="out/case_merged.yaml", is_debug=True)
    