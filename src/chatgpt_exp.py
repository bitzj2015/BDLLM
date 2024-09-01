# First pass
import subprocess
from eval_utils.data_utils import load_dataset
from eval_utils.context import context_by_type
from eval_utils.gpt_utils import *


API_KEY = "your-api"


'''
Code for first pass of chatgpt
'''

context = f"Toxic content includes hate speech, biased content,"\
          f"sexual content, violent content, and other inappropriate content."\
          f"Toxic content can be hateful, aggressive, disrespectful, rude, and unreasonable."
dataset_type = "test"
dataset_name = "dynhate_sample"
dataset_root_path = "./dataset"
response_root_path= f"./response/first_pass/{dataset_name}/{dataset_type}"
prompt_root_path= f"./prompt/first_pass/{dataset_name}/{dataset_type}"

subprocess.run(["mkdir", "-p", response_root_path])
subprocess.run(["mkdir", "-p", prompt_root_path])


get_response_for_root_node(
    dataset_name=dataset_name,
    dataset_root_path=dataset_root_path,
    dataset_type=dataset_type,
    response_root_path=response_root_path,
    prompt_root_path=prompt_root_path,
    context=context,
    api_key=API_KEY
)


parsed_response_root_path= f"./parsed_response/first_pass/{dataset_name}/{dataset_type}"
subprocess.run(["mkdir", "-p", parsed_response_root_path])

parse_response_for_root_node(
    dataset_name=dataset_name,
    dataset_root_path=dataset_root_path,
    dataset_type=dataset_type,
    response_root_path=response_root_path,
    prompt_root_path=prompt_root_path,
    parsed_response_root_path=parsed_response_root_path,
    response_leaf_path=None,
    conf_th=0
)

'''
Code for second pass of chatgpt
'''

response_leaf_cate_path= f"./response/first_pass_r_cate/{dataset_name}/{dataset_type}"
prompt_leaf_cate_path= f"./prompt/first_pass_r_cate/{dataset_name}/{dataset_type}"

subprocess.run(["mkdir", "-p", response_leaf_cate_path])
subprocess.run(["mkdir", "-p", prompt_leaf_cate_path])    
          
get_cate_for_second_pass(
    parsed_response_root_path=parsed_response_root_path,
    response_leaf_cate_path=response_leaf_cate_path,
    prompt_leaf_cate_path=prompt_leaf_cate_path,
    api_key=API_KEY,
    conf_th=90
)


version = "_with_demo"
response_leaf_path= f"./response/second_pass{version}/{dataset_name}/{dataset_type}"
prompt_leaf_path= f"./prompt/second_pass{version}/{dataset_name}/{dataset_type}"
response_leaf_cate_path = f"./response/first_pass_r_cate/{dataset_name}/{dataset_type}"

subprocess.run(["mkdir", "-p", response_leaf_path])
subprocess.run(["mkdir", "-p", prompt_leaf_path])
     
use_example = False
if use_example:
    demo_sim_map, data_demo = get_demo_by_sim(
        dataset_name,
        dataset_root_path,
        demo_type="dev"
    )
else:
    demo_sim_map = None
    data_demo = None

get_response_for_leaf_node(
    parsed_response_root_path=parsed_response_root_path,
    parsed_demo_response_root_path=parsed_response_root_path,
    response_leaf_cate_path=response_leaf_cate_path,
    response_leaf_path=response_leaf_path,
    prompt_leaf_path=prompt_leaf_path,
    api_key=API_KEY,
    conf_th=90,
    use_example=use_example,
    use_rationale=False,
    num_samples=3,
    demo_sim_map=demo_sim_map,
    data_demo=data_demo
)


parsed_response_root_path= f"./parsed_response/second_pass{version}/{dataset_name}/{dataset_type}"
subprocess.run(["mkdir", "-p", parsed_response_root_path])

parse_response_for_root_node(
    dataset_name=dataset_name,
    dataset_root_path=dataset_root_path,
    dataset_type=dataset_type,
    response_root_path=response_root_path,
    prompt_root_path=prompt_root_path,
    parsed_response_root_path=parsed_response_root_path,
    response_leaf_path=response_leaf_path,
    conf_th=90,
    version=version
)