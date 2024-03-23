import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
from clrnet.datasets import build_dataloader
from clrnet.utils.visualization import imshow_lanes


from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

# from mmcv.parallel import DataContainer as DC


from ultralytics import YOLO
import json
import yaml
import copy

from pprint import pprint


img_w = 800
img_h = 320


def convert_dict_to_list(dictionary):
    result = []
    keys = list(dictionary.keys())
    values = dictionary[keys[0]]
    num_items = len(values)

    for i in range(num_items):
        item = {}
        for key in keys:
            item[key] = dictionary[key][i]
        result.append(item)

    return result


# # Example usage
# input_dict = {"key1": ['item1', 'item2', 'item3']}
# output_list = convert_dict_to_list(input_dict)
# print(output_list)


class MyImageDataset(Dataset):
    """
    Image dataset for inference on new data
    """

    def __init__(self, image_list_file, cfg):
        self.cfg = cfg
        self.image_list = []
        with open(image_list_file, "r") as file:
            for line in file:
                image_path = line.strip()
                if os.path.isfile(image_path):
                    self.image_list.append(image_path)

        # print("========== In ImageDataset ==========")
        # print("self.image_list: ", end="")
        # print(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Customized __getitem__ method to load images from the list
        """
        image_path = self.image_list[index]
        image = cv2.imread(image_path)
        # image = image[self.cfg.cut_height:, :, :]
        image = image.astype(np.float32) / 255.0

        # print("========== In ImageDataset.__getitem__ ==========")
        # print(f"image_path: {image_path}")
        # print(f"image.shape: {image.shape}")

        image = cv2.resize(image, (img_w, img_h))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = torch.from_numpy(image.transpose((2, 0, 1)))  # Transpose dimensions

        sample = {}
        sample.update({"img": image})

        meta = {"img_name": image_path.split("/")[-1], "img_root": image_path}
        # meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})
        return sample
        # return image

    def view(self, predictions, img_metas):
        """
        Originated from `view` in `clrnet/datasets/base_dataset.py`
        """
        # img_metas = [item for img_meta in img_metas.data for item in img_meta]
        # img_metas = [item for img_meta in img_metas for item in img_meta]

        img_metas = convert_dict_to_list(img_metas)

        # print("========== In ImageDataset.view ==========")
        # print(f"img_metas: {img_metas}")
        # input("Press Enter to continue...")

        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta["img_name"]
            img = cv2.imread(img_meta["img_root"])
            out_file = os.path.join(
                self.cfg.work_dir, "visualization", img_name.replace("/", "_")
            )

            print("========== In ImageDataset.view ==========")
            print(f"out_file: {out_file}")

            input("Press Enter to continue...")

            pred = copy.deepcopy(lanes)

            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

            output = self.get_prediction_string(pred)
            with open(out_file[:-4] + ".txt", "w") as f:
                f.write(output)

    def get_prediction_string(self, pred):
        """
        Originated from `get_prediction_string` in `clrnet/datasets/culane.py`
        """
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = " ".join(
                ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
            )
            if lane_str != "":
                out.append(lane_str)

        return "\n".join(out)


class MyRunner(Runner):
    """
    To infer the lane detection results on customized images
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def infer(self):
        """
        To infer the lane detection results on customized images
        """
        # Example usage
        image_list_file = ...  # The file containing the list of images to be inferred
        batch_size = 24
        num_workers = 4

        dataset = MyImageDataset(image_list_file, self.cfg)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=False,
            drop_last=False,
            # collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            # worker_init_fn=init_fn
        )

        # for batch in dataloader:
        #     # Use the batch of images for training or evaluation
        #     print(batch.shape)

        self.test_loader = dataloader

        # Original code
        # if not self.test_loader:
        #     self.test_loader = build_dataloader(self.cfg.dataset.test,
        #                                         self.cfg,
        #                                         is_train=False)

        # print("==========")
        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)
        #     break

        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f"Testing")):
            data = self.to_cuda(data)

            # print("========== In MyRunner.infer() ==========")
            # print(data['img'].shape)
            # print(type(data))
            # input("Press Enter to continue...")

            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data["meta"])

        # metric = self.test_loader.dataset.evaluate(predictions, self.cfg.work_dir)
        # if metric is not None:
        #     self.recorder.logger.info("metric: " + str(metric))


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    # runner = Runner(cfg)
    runner = MyRunner(cfg)

    runner.infer()

    # if args.validate:
    #     runner.validate()
    # elif args.test:
    #     runner.test()
    # else:
    #     runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dirs", type=str, default=None, help="work dirs")
    parser.add_argument(
        "--load_from", default=None, help="the checkpoint file to load from"
    )
    parser.add_argument(
        "--resume_from", default=None, help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--finetune_from", default=None, help="the checkpoint file to resume from"
    )
    parser.add_argument("--view", action="store_true", help="whether to view")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to test the checkpoint on testing set",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default="0")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    return args


def gen_img_list(img_dir_path, fp_save):
    """
    Generate a list of images in the given directory
    
    Parameters
    ----------
    img_dir_path : str
        The path to the directory containing the images
    fp_save : str
        The path to the file where the list of images will be saved
    """

    ls_img_dir = os.listdir(img_dir_path)
    ls_img_dir.sort()

    ls_ans = []
    for img_fn in ls_img_dir:
        local_path = os.path.join(img_dir_path, img_fn)
        print(local_path)
        if img_fn.endswith(".jpg"):
            ls_ans.append(local_path)
        # ls_files = os.listdir(local_path)
        # for y in ls_files:
        #     if y.endswith(".jpg"):
        #         ls_ans.append(local_path+"/"+y)

    print(ls_ans)
    with open(fp_save, "w") as f:
        for img_fn in ls_ans:
            f.write(img_fn + "\n")


if __name__ == "__main__":
    gen_img_list(img_dir_path, file_img_list)

    main()
