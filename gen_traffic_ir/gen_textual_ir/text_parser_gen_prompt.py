def gen_prompt(traffic_description):
    prompts = []

    instruction = """
I want you to act as a test expert for autonomous driving systems. Your task is to derive test scenarios from given test scenario description following the below step-by-step instructions.

Step 1 - Derive the weather element of the test scenario. The weather could be one of the option in [sunny, foggy, rainy, snowly, cloudy]. Output the weather element. If the weather does not explicitly mentioned in the test scenario description, you can set it as the default weather "sunny". If no options can describe the weather mentioned in the test scenario description, create a new option by yourself.

Step 2 - Derive the road network element(s) of the test scenario. The road network elements contain road network topologies, traffic signs or signals, road markers, and road types where the scenario occurs. The options of road network topologies could be [intersection, t-intersection, roundabout]. The options of traffic signs and signals could be [traffic light, stop sign, speed limit sign]. The options of road markers could be [solid line, broken line]. The options of road types could be [one way road, two lane road]. If no any road network topology, traffic signs and signals, road markers or road types are explicitly mentioned in the test scenario description, you can set road network element as "None".  
Output all mentioned road network elements as a list in []. If no options can describe the road network elements mentioned in the test scenario description, create a new option by yourself.

Step 3 - Derive whether there are other actors besides the ego vehicle in the test scenario description. The options of other actors could be [car, pedestrian]. Output all possible actors in a list [] if they are explicitly or implicitly mentioned in the test scenario description. If no other actors mentioned in the test scenario description, output "None". If no options can describe the actors mentioned in the test scenario description, create a new option by yourself.

Step 4 - Derive the current behavior of the ego vehicle. The options are [go forward, yield, stop, turn left, turn right, change lane to left, change lane to right]. If the behavior of the ego vehicle is not mentioned in the test scenario description, set it as "go forward". If no options can describe the actors mentioned in the test scenario description, create a new option by yourself. If the speed is indicated, output the speed in the format of "speed: 30 mph" or "speed: 30 km/h". If the speed is not indicated, output "None".

Step 5 - Derive the position of the ego vehicle when the test scenario occurs. The position contains two part "Position target" and "Position relation". “position target” is one of the reference road network element obtained in Step 2 and “position relation” describes the relative position between the ego vehicle and the reference position target. The options are [front, behind, left, right, opposite, left behind, right behind]. For example, if the "position target" is "intersection" and the "position relation" is "behind", it means the ego vehicle is behind (approaching) the intersection. If no road work elements are obtained in Step 2, output [None , None], otherwise output Output position target and position relation in a list.

Step 6 - Derive the intended behavior of the other actor following the same process as Step 4.

Step 7 - Derive the position of the other actor following the similar process as Step 5. The different point is that the "position target" could be either "ego vehicle" or one of the obtained road network elements based on the description in the test scenario description.

Step 8 - Save all elements obtained from Steps 1-7 to a YAML, following the below syntax:
YAML:
road_network:
weather:
participant:
    ego_vehicle:
        current_behavior:
        position_target:
        position_relation:
        speed:
    other_actor_1:
        type:
        current_behavior:
        position_target:
        position_relation:
        speed:
traffic_signal:
EOF
Answer the question with only the YAML format, starting with the keyword ``YAML:'' and ending with ``EOF''.
    """

    example_scenario_description = """
It is currently daytime with partly cloudy skies and clear visibility.
Our ego vehicle is driving on the third lane from the left on a five-lane road at the speed of 30 mph. 
There are pedestrians crossing at the intersection directly ahead of it. 
There is a black sedan in front of the ego vehicle in the right lane, going forward at the speed of 45 mph.
There are cars driving in front of the ego vehicle in both left two lanes, going forward at the speed of 35 mph.  
The traffic light for the direction of the ego vehicle is currently green.
    """

    example_output = """
YAML:
road_network: intersection
weather: partly cloudy
participant: 
    ego_vehicle:
        current_behavior: go forward
        position_target: intersection
        position_relation: front
        speed: 30 mph
    other_actor_1:
        type: pedestrian
        current_behavior: crossing
        position_target: intersection
        position_relation: front
        speed: None
    other_actor_2:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: right front
        speed: 45 mph
    other_actor_3:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: left front
        speed: 35 mph
traffic_signal: green
EOF
    """

    example_scenario_description_2 = """
It is currently daytime and sunny with clear visibility.
Our ego vehicle is driving on the third lane from the left on a five-lane road. 
There are pedestrians crossing at the intersection directly ahead of it. 
There is a white sedan in front of the ego vehicle in the right lane, going forward.
There are cars driving in front of the ego vehicle in both left two lanes, going forward.  
There is a stop sign at the intersection.
    """

    example_output_2 = """
YAML:
road_network: intersection
weather: sunny
participant: 
    ego_vehicle:
        current_behavior: go forward
        position_target: intersection
        position_relation: front
        speed: None
    other_actor_1:
        type: pedestrian
        current_behavior: crossing
        position_target: intersection
        position_relation: front
        speed: None
    other_actor_2:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: right front
        speed: None
    other_actor_3:
        type: car
        current_behavior: go forward
        position_target: ego vehicle
        position_relation: left front
        speed: None
traffic_signal: stop_sign
EOF
    """

    prompts.append({"role": "system", "content": instruction})
    prompts.append({"role": "user", "content": example_scenario_description})
    prompts.append({"role": "assistant", "content": example_output})
    prompts.append({"role": "user", "content": example_scenario_description_2})
    prompts.append({"role": "assistant", "content": example_output_2})
    prompts.append({"role": "user", "content": traffic_description})

    return prompts
