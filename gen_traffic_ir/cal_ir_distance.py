import os

import difflib
import deepdiff
import jsondiff
import yaml
import json
import re

from pprint import pprint


class IrDistanceCalculator:
    def __init__(self):
        # TODO: User needs to specify the following paths to the dir
        self.dir_ground_truth = ""
        self.dir_gpt = ""
        self.dir_7b = ""
        self.dir_13b = ""
        self.dir_70b = ""
        self.report_file = ""

    @staticmethod
    def get_file_pairs(dir1, dir2):
        files_dir1 = set(os.listdir(dir1))
        files_dir2 = set(os.listdir(dir2))
        common_files = files_dir1.intersection(files_dir2)
        return [
            (os.path.join(dir1, file), os.path.join(dir2, file))
            for file in common_files
        ]

    def cal_dist_json(self, file_ori, file2, is_debug=False):
        with open(file_ori, "r") as f_ori, open(file2, "r") as f2:
            dc_ori = yaml.load(f_ori.read().lower(), Loader=yaml.FullLoader)
            dc2 = yaml.load(f2.read().lower(), Loader=yaml.FullLoader)

        # ## calculate the total number of items in the json file
        counter = 0
        for k, v in dc_ori.items():
            if isinstance(v, str):
                # ## count the number of level 1 attributes such as the road network, weather, time of day, etc.
                counter += 1
            elif isinstance(v, dict):
                # ## count the number of level 2 attributes for each level 1 attribute with a dictionary, 
                # ## such as the position of the vehicle, the type of the vehicle, the current behavior of the vehicle, etc.
                for k2, v2 in v.items():
                    counter += len(v2)
            elif isinstance(v, list):
                counter += len(v)
            else:
                print(type(v))
                raise ValueError("not string or dict or list")

        counter_diff = 0

        # ## deepdiff
        ans = deepdiff.DeepDiff(dc_ori, dc2)

        print(f"counter: {counter}")
        print(f"dc1:\n{dc_ori}")
        print()
        print(f"dc2:\n{dc2}")
        print()
        print(f"diff:\n{ans}")
        print()
        # NOTE: There will be in total 3 possible types of differences between two json files:
        # 1. dictionary_item_added
        # 2. dictionary_item_removed
        # 3. values_changed
        for key in [
            "dictionary_item_added",
            "dictionary_item_removed",
            "values_changed",
        ]:
            if key in ["dictionary_item_added", "dictionary_item_removed"]:
                if key in ans:
                    counter_diff += len(ans[key])
                else:
                    if is_debug:
                        print("dictionary_item_added not in ans")
            else:
                if key in ans:
                    dc_diff = ans[key]
                    num_exclude = self.apply_exclude_rules(dc_diff)

                    if is_debug:
                        print(key)
                        print(len(dc_diff.keys()))

                        print(f"before counter_diff: {counter_diff}")
                        print(f"num_exclude: {num_exclude}")
                        print(f"len(ans[key]): {len(ans[key])}")
                    counter_diff = counter_diff + len(ans[key]) - num_exclude

                    if is_debug:
                        print(f"coutner_diff: {counter_diff}")
                else:
                    if is_debug:
                        print("WARNING: values_changed not in ans")

        diff_ratio = counter_diff
        print(f"diff_ratio: {diff_ratio:.4f}")

        # ## jsondiff
        # NOTE: jsondiff does not explicitly show the "added", "removed", "changed" items between two json files
        # ans = jsondiff.diff(dc_ori, dc2)
        # print("jsondiff")
        # print(ans)
        # print(f"type: {type(ans)}")
        # print(f"ans.keys(): {ans.keys()}")
        # for k, v in ans.items():
        #     print(f"k: {k}")
        #     print(f"v: {v}")
        #     print(type(v))
        #     print()

        return diff_ratio

    def apply_exclude_rules(self, dc_diff):
        """
        Apply the exclude rules to the json file
        return: the number of items that are excluded
        """
        ls_car_synonyms = ["car", "vehicle"]
        ls_driving_synonyms = ["go forward", "go straight", "go", "drive", "driving"]
        ls_speed_synonyms = [
            "none",
            "0 mph",
            "0 miles per hour",
            "0 miles/hour",
            "0 miles/hr",
            "0 km/h",
            "0 kmh",
            "0 kmph",
            "0 kilometers per hour",
            "0",
        ]
        counter_exclude = 0

        for k, v in dc_diff.items():
            # if v['new_value'] == None or v['old_value'] == None:
            #     continue
            # TODO: add more conditions to filter out the irrelevant items
            if k.startswith("root['participant']"):
                if k.endswith("['type']"):
                    if (
                        v["new_value"] in ls_car_synonyms
                        and v["old_value"] in ls_car_synonyms
                    ):
                        counter_exclude += 1
                        print("exclude")
                        print(f"k: {k}")
                        print(f"v: {v}")
                if k.endswith("['current_behavior']"):
                    if (
                        v["new_value"] in ls_driving_synonyms
                        and v["old_value"] in ls_driving_synonyms
                    ):
                        print("match current_behavior")
                        print(f"k: {k}")
                        print(f"v: {v}")
                        counter_exclude += 1
                        print("exclude")
                if k.endswith("['speed']"):
                    if (
                        v["new_value"] in ls_speed_synonyms
                        and v["old_value"] in ls_speed_synonyms
                    ):
                        print("match speed")
                        print(f"k: {k}")
                        print(f"v: {v}")
                        counter_exclude += 1
                        print("exclude")
        return counter_exclude

    def generate_diff_report(self):
        """Generate a diff report between two directories containing json files"""
        report_file = self.report_file
        file_pairs = self.get_file_pairs(self.dir_gpt, self.dir_70b)
        with open(report_file, "w") as report:
            for file1, file2 in file_pairs:
                print()
                print("==========")
                print(file1)
                # diffs = self.cal_dist_json(file1, file2)
                diff_ratio = self.cal_dist_json(file2, file1)
                # print(diffs)
                # if diffs:
                #     report.write("\n".join(diffs) + "\n\n")
                # break


if __name__ == "__main__":
    runner = IrDistanceCalculator()
    runner.generate_diff_report()
