import openai
from openai import OpenAI
import time
import json
import pickle
import os
from tqdm import tqdm
import re

from dotenv import load_dotenv


from gen_traffic_ir.gen_textual_ir.text_parser_gen_prompt import gen_prompt


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_scenario_all_in_one(traffic_description):
    prompts = gen_prompt(traffic_description)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=prompts,
    )

    # reply = response["choices"][0]["message"]["content"]
    reply = response.choices[0].message.content

    return reply


def post_process(content):
    """
    Extracts content from a string that is between the keywords 'yaml:' and 'end_yaml'.
    """
    
    pattern = r"YAML:(.*?)EOF"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        ans = match.group(1).strip()
        return ans
    else:
        return None


def run(is_debug=False):
    dir_path: str
    # The path of the folder containing the scenario description files

    save_dir: str

    ls_txt = os.listdir(dir_path)
    ls_txt.sort()
    ls_txt = [os.path.join(dir_path, i) for i in ls_txt if i.endswith(".txt")]

    answers = []
    for desc_file_path in tqdm(ls_txt):
        with open(desc_file_path, "r") as scenario_file:
            scenario = scenario_file.readlines()
        scenario = "".join(scenario)
        if scenario == "":
            print(f"file {desc_file_path} is empty")
            continue

        if is_debug:
            print(scenario)
            print()

        scenario_info = parse_scenario_all_in_one(scenario)

        while True:
            try:
                scenario_info = post_process(scenario_info)
                # reply = parse_scenario_all_in_one(scenario=scenario)
                
                print(f"Output:")
                print(scenario_info)
                print()
                answers.append(scenario_info)
                time.sleep(5)
                print()

                save_file_name = os.path.basename(desc_file_path)
                if save_file_name.endswith(".txt"):
                    save_file_name = save_file_name[:-4] + ".yaml"
                save_path = os.path.join(save_dir, save_file_name)
                with open(save_path, "w") as fp:
                    # json.dump(scenario_info, fp)
                    fp.write(scenario_info)

                if is_debug:
                    exit()
                break
            except:
                time.sleep(30)

        time.sleep(5)

    # # filehandler = open('scenarios_all_in_one.pkl', 'wb')
    # # pickle.dump(answers, filehandler)


if __name__ == "__main__":
    run(is_debug=False)
