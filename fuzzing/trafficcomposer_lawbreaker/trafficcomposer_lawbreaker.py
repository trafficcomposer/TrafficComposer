import yaml


def generate_bnf(yaml_content):
    bnf = """
<AVUnit> ::= <lane_num> <participants> <road_network> <weather>

<lane_num> ::= "lane_num: " <number>
<participants> ::= <ego_vehicle> <other_actors>
<ego_vehicle> ::= "participant:\\n    ego_vehicle:\\n" <vehicle_attributes>
<other_actors> ::= <other_actor> | <other_actor> <other_actors>
<other_actor> ::= "    other_actor_" <number> ":\\n" <actor_attributes>
<vehicle_attributes> ::= <current_behavior> <lane_idx> <position_relation> <position_target> <speed>
<actor_attributes> ::= <vehicle_attributes> <type>

<current_behavior> ::= "        current_behavior: " <behavior> "\\n"
<lane_idx> ::= "        lane_idx: " <lane_index> "\\n"
<position_relation> ::= "        position_relation: " <relation> "\\n"
<position_target> ::= "        position_target: " <target> "\\n"
<speed> ::= "        speed: " <speed_value> "\\n"
<type> ::= "        type: " <actor_type> "\\n"

<behavior> ::= "go forward" | "crossing"
<lane_index> ::= <number> | "None"
<relation> ::= "front" | "right"
<target> ::= "intersection" | "ego vehicle"
<speed_value> ::= "None"
<actor_type> ::= "pedestrian" | "car"
<road_network> ::= "road_network: [" <traffic_control> "]"
<traffic_control> ::= "traffic light"
<weather> ::= "weather: " <weather_condition>
<weather_condition> ::= "sunny"

<number> ::= <digit> | <digit> <number>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    """
    return bnf.strip()


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_bnf(file_path, bnf_content):
    with open(file_path, "w") as file:
        file.write(bnf_content)


if __name__ == "__main__":
    input_file = "input.yaml"  # Update this with the path to your YAML file
    output_file = "output.bnf"

    # Read YAML content
    yaml_content = read_yaml(input_file)

    # Generate BNF content based on the YAML structure
    bnf_content = generate_bnf(yaml_content)

    # Write BNF content to a file
    write_bnf(output_file, bnf_content)

    print(f"BNF content has been written to {output_file}")
