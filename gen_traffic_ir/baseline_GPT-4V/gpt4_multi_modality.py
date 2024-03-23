import openai
import time
import json
import pickle
import os
import cv2
import base64
import requests
import textwrap
import re

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def resize_img(image, target_long_side=500):  # Initialize the target dimensions
    # Load the image
    # image_path = "path_to_your_image.jpg"
    # image = cv2.imread(image_path)

    # Get original dimensions
    height, width = image.shape[:2]

    if height < target_long_side and width < target_long_side:
        return image

    # Calculate the ratio of the new dimension to the old dimension
    if height > width:
        ratio = target_long_side / height
        new_height = target_long_side
        new_width = int(width * ratio)
    else:
        ratio = target_long_side / width
        new_width = target_long_side
        new_height = int(height * ratio)

    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    # Save or display the resized image
    # cv2.imshow("Resized Image", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optionally, save the image
    # cv2.imwrite('resized_image.jpg', resized_image)
    return(resized_image)


def run_gpt4v_gen_ir():
    img_dir = "multi-modality-auto-drive/05191714_0507.MP4" # correct for image direction
    description_dir = (
        "multi-modality-auto-drive/gen_merged_ir/data/description_v2"   # correct for data description 
    )

    ls_desc = os.listdir(description_dir)
    ls_desc.sort()
    ls_desc = [i.strip(".txt") for i in ls_desc if i.endswith(".txt")]
    for file_name in ls_desc:
        img_fp = os.path.join(img_dir, file_name + ".jpg")
        desc_fp = os.path.join(description_dir, file_name + ".txt")

        if not os.path.exists(img_fp):
            print(f"file {img_fp} does not exist")

        if not os.path.exists(desc_fp):
            print(f"file {desc_fp} does not exist")

        with open(desc_fp, "r") as fp:
            desc = fp.read()

        img = cv2.imread(img_fp)
        img = resize_img(img, target_long_side=500)
        # img = cv2.resize(img, (150, 150))
        cv2.imwrite(img_fp, img)
        

        gpt4v_gen_ir(image_path=img_fp, desc=desc)


def gpt4v_gen_ir(image_path, desc, iter = 0):
    if iter > 3:
        print("OpenAI Error while generating " + image_path.split(".")[1])
        return

    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image
    # image_path = "path_to_your_image.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompts = []

    instruction = """
I want you to act as a test expert for autonomous driving systems. Your task is to derive test scenarios from the given test scenario description and image following the below step-by-step instructions. Do not provide any output other than the YAML file that is required to be generated after all steps are completed.

Step 1 - Derive the weather element of the test scenario. The weather could be one of the options in [sunny, foggy, rainy, snowly]. Output the weather element. If the weather is not explicitly mentioned in the test scenario description, you can set it as the default weather "sunny". If no options can describe the weather mentioned in the test scenario description, create a new option by yourself.

Step 2 - Derive the road network element(s) of the test scenario. The road network elements contain road network topologies, traffic signs or signals, road markers, and road types where the scenario occurs. The options of road network topologies could be [intersection, t-intersection, roundabout]. The options of traffic signs and signals could be [traffic light, stop sign, speed limit sign]. The options for road markers could be [solid line, broken line]. The options of road types could be [one way road, two lane road]. If no road network topology, traffic signs and signals, road markers, or road types are explicitly mentioned in the test scenario description, you can set the road network element as "None". Output all mentioned road network elements as a list in []. If no options can describe the road network elements mentioned in the test scenario description, create a new option by yourself.

Step 3 - Derive whether there are other actors besides the ego vehicle in the test scenario description. The options of other actors could be [car, pedestrian, train]. Output all possible actors in a list [] if they are explicitly or implicitly mentioned in the test scenario description. If no other actors are mentioned in the test scenario description, output "None". If no options can describe the actors mentioned in the test scenario description, create a new option by yourself.

Step 4 - Derive the current behavior of the ego vehicle. The options are [go forward, yield, stop, turn left, turn right, change lane to left, change lane to right]. If the behavior of the ego vehicle is not mentioned in the test scenario description, set it as "go forward". If no options can describe the actors mentioned in the test scenario description, create a new option by yourself. If the speed is indicated, output the speed in the format of "speed: 30 mph" or "speed: 30 km/h". If the speed is not indicated, output "None".

Step 5 - Derive the position of the ego vehicle when the test scenario occurs. The position contains two parts: "Position target" and "Position relation". "position target" is one of the reference road network elements obtained in Step 2, and "Position relation" describes the relative position between the ego vehicle and the reference position target. The options are [front, behind, left, right, opposite, left behind, right behind]. For example, if the "position target" is "intersection" and the "position relation" is "behind", it means the ego vehicle is behind (approaching) the intersection. If no road work elements are obtained in Step 2, output [None, None]; otherwise, output Output position target and position relation in a list.

Step 6 - Derive the intended behavior of the other actor following the same process as Step 4.

Step 7 - Derive the position of the other actor following a similar process as Step 5. The different point is that the "position target" could be either "ego vehicle" or one of the obtained road network elements based on the description in the test scenario description.

Step 8 - Detect each lane in the image and output the total lane number and the lane number of the ego vehicle. If the ego vehicle is not in the lane, output "None".

Step 9 - Detect each vehicle and pedestrian in each lane and match the detected vehicles and pedestrians with the corresponding ones from the test scenario description. If the vehicle or pedestrian is not mentioned in the test scenario description, output "None".

Step 10 - Save all elements obtained from Steps 1-9 to a YAML following the below syntax:

{
    road_network:
    weather:
    lane_num:
    participant:
        ego_vehicle:
            current_behavior:
            lane_idx:
            position_relation:
            position_target:
            speed:
        other_actor_1:
            current_behavior:
            lane_idx:
            position_relation:
            position_target:
            speed:
            type:
    traffic_signal:
}
    """

    example_scenario_description = """
It is currently daytime with partly cloudy skies and clear visibility.
Our ego vehicle is driving on the third lane from the left on a five-lane road at the speed of 30 mph. 
There are pedestrians crossing at the intersection directly ahead of it. 
There is a black sedan in front of the ego vehicle in the right lane, going forward at the speed of 45 mph.
There are cars driving in front of the ego vehicle in both left two lanes, going forward at the speed of 35 mph.  
The traffic light for the direction of the ego vehicle is currently green.
    """
    example_img_path = "gpt4v_eg1.jpg"
    base64_image_example = encode_image(example_img_path)
    example_aligned_gt = """
lane_num: 5
participant:
    ego_vehicle:
        current_behavior: go forward
        lane_idx: 3
        position_relation: front
        position_target: intersection
        speed: 30 mph
    other_actor_1:
        current_behavior: go forward
        lane_idx: 4
        position_relation: front
        position_target: intersection
        speed: 45 mph
        type: car
    other_actor_2:
        current_behavior: go forward
        lane_idx: 1
        position_relation: None
        position_target: None
        speed: 35 mph
        type: pedestrian
    other_actor_3:
        current_behavior: go forward
        lane_idx: 2
        position_relation: front
        position_target: intersection
        speed: 35 mph
        type: car
    other_actor_4:
        current_behavior: go forward
        lane_idx: 2
        position_relation: front
        position_target: intersection
        speed: 35 mph
        type: car
road_network:
    - two lane road
    - traffic light
weather: partly cloudy
"""

    example_scenario_description_2 = """
It is currently daytime with partly cloudy skies and clear visibility.
Our ego vehicle is driving on the third lane from the left on a five-lane road at the speed of 30 mph. 
There are pedestrians crossing at the intersection directly ahead of it. 
There is a black sedan in front of the ego vehicle in the right lane, going forward at the speed of 45 mph.
There are cars driving in front of the ego vehicle in both left two lanes, going forward at the speed of 35 mph.  
The traffic light for the direction of the ego vehicle is currently green.
    """
    example_img_path_2 = (
        "gpt4v_eg2.png"
    )
    base64_image_example_2 = encode_image(example_img_path_2)
    example_aligned_gt_2 = """
"""

    prompts.append({"role": "system", "content": instruction})
    prompts.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": example_scenario_description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_example}"
                    },
                },
            ],
        }
    )
    prompts.append({"role": "assistant", "content": example_aligned_gt})
    prompts.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": example_scenario_description_2},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_example_2}"
                    },
                },
            ],
        }
    )
    prompts.append({"role": "assistant", "content": example_aligned_gt_2})
    prompts.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": desc},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    )

    payload = {
        "model": "gpt-4-vision-preview",
        # "messages": [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": prompt},
        #             {
        #                 "type": "image_url",
        #                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        #             },
        #         ],
        #     }
        # ],
        "messages": prompts,
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    time.sleep(5)

    response_dic: dict = response.json()

    if "error" in response_dic:
        time.sleep(30)
        gpt4v_gen_ir(image_path=image_path, desc=desc, iter = iter + 1)
    else:
        f = open(
            os.getcwd() + "/gpt-v-seed/" + image_path.split('/')[-1][:-4]+ ".yaml",
            "w",
        )
        print(str(response_dic))
        try:
            outputstr = (
                response_dic.get("choices")[0]
                .get("message")
                .get("content")
                .split("{")[1]
                .split("}")[0]
            )
            outputstr = "\n".join(textwrap.dedent(outputstr).split("\n")[1:])
            f.write(outputstr)

        except:
            try:
                outputstr = (
                    response_dic.get("choices")[0]
                    .get("message")
                    .get("content")
                )
                outputstr = "\n".join(textwrap.dedent(outputstr).split("\n")[1:])
                f.write(outputstr)
            except:
                f.write(str(response_dic))
        f.close()


if __name__ == "__main__":
    run_gpt4v_gen_ir()
