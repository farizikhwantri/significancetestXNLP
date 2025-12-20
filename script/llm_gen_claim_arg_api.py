import os
import time
import random
import argparse

import logging

from collections import defaultdict

import json
import pandas as pd

from llm_gen_claim_arg import sac_structure, sac_cae_json_schema, construct_prompt
from utils_prompt import construct_prompt_from_template

from llm_sikt_chat import gpt_request


def main(args):
    # Initialize the model and tokenizer
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info('args: %s', args)

    def_model_name = "gpt-4o"
    model_name = args.model_name if args.model_name is not None else def_model_name

    # sys_msg = """You are a security assurance engineer. Your duty is to produce a thorough Security Assurance Case
    # from IEC 62443 requirements. {sac_structure}""".format(sac_structure=sac_structure)

    # def_sys_msg = """You are a legal expert in privacy and security issue. Your duty is to produce a thorough Data Processing Agreement Assurance Case
    # from General Data Protection Regulation legal requirements. {sac_structure}""".format(sac_structure=sac_structure)


    def_sys_msg = """You are a legal expert in privacy and security issue. Your duty is to produce a thorough Security Assurance Case
    from Cybersecurity Resilience Act requirements. {sac_structure}""".format(sac_structure=sac_structure)

    # sys_msg = args.system_msg if args.system_msg is not None else def_sys_msg

    # read arguments system message from args of json file
    if args.system_msg is not None:
        with open(args.system_msg, 'r') as f:
            # write a function to read the system message according to file type e.g. json, txt, etc.
            if args.system_msg.endswith('.json'):
                sys_msg = json.load(f).get('system_message', def_sys_msg)['content']
                sys_msg = sys_msg.strip() if sys_msg else def_sys_msg
            elif args.system_msg.endswith('.txt'):
                sys_msg = f.read().strip()
    else:
        # use default system message
        sys_msg = def_sys_msg
    
    # If the txt file doesn't contain a system message, use the default
    if not sys_msg:
        sys_msg = def_sys_msg


    results_dict = defaultdict(dict)
    ncalls = args.ncalls

    # load the input file
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.json'):
        df = pd.read_json(args.input_file)
    else:
        raise ValueError("Input file should be either csv or json")
    
    # count failed json
    failed_json = 0

    # format_prompt = f"such a way that the data matches the follwing schema: {sac_cae_json_schema}. Give just the json without any explanation. \
    #     Follow the JSON schema key convention naming. Do not copy the schema or obvious example. Print the json in a single line, do not write a new line."

    format_prompt = f"such a way that the data matches the follwing schema: {sac_cae_json_schema}. Give just the json without any explanation."
    
    if args.user_prompt is not None:
        # read the user prompt from the file
        with open(args.user_prompt, 'r') as f:
            user_prompt = f.read().strip()
        format_prompt = f"{user_prompt}"

    for i, row in df.iterrows():
        # row_id = row['requirement_id']
        row_id = row[args.row_id] if args.row_id in row else row['requirement_id']
        prompt = None
        if args.prompt_template_path is not None:
            # construct prompt from template
            prompt = construct_prompt_from_template(args.prompt_template_path, 
                                                    **row.to_dict(), format=format_prompt)
            print(f"Constructed prompt for requirement {row_id}: {prompt}")
        else:
            req_name = row['requirement_name']
            req_desc = row['requirement_description']
            # req_rationale = row['requirement_rationale']
            req_rationale = row['manual_rationale'] if 'manual_rationale' in row else row['requirement_rationale']

            results_dict[row['requirement_id']] = defaultdict(list)

            prompt = construct_prompt(row_id, req_name, req_desc, req_rationale, 
                                      format_prompt)

        # messages = [
        #     {"role": "system", "content": sys_msg},
        #     {"role": "user", "content": prompt},
        # ]
        # if not sys_role_flag:
        #     # contate system message to the prompt
        #     prompt = sys_msg + ' ' + prompt
        #     messages = [
        #         {"role": "user", "content": prompt},
        #     ]

        print(f"Processing requirement {row_id}")

        for i in range(ncalls):
            model_base_name = model_name.split('/')[-1]

            # check if "{args.output_dir}/{model_base_name}_{req_id}_{i}.json" exists
            # if it exists, skip the call
            if os.path.exists(f"{args.output_dir}/{model_base_name}_{row_id}_{i}.json"):
                print(f"Skipping requirement {row_id} in call {i}")
                continue
            
            response = gpt_request(session_id=args.session_id, 
                                   message=prompt, prompt=sys_msg, 
                                   model_id=args.model_id, 
                                   model_name=model_name, 
                                   temperature=args.temperature,
                                   max_length=args.max_length,
                                   token_limit=args.token_limit,
                                   chatmode=False)

            logger.info(f"Generated text for requirement {row_id} in call {i}: {response}")

            if i % ncalls == 0:
                print(f"Processed {i} requirements")

            print(f"Generated text for requirement {row_id} in call {i}: {response}")

            # save the results to a json file
            with open(f"{args.output_dir}/{model_base_name}_{row_id}_{i}.json", "w") as file:
                json.dump(response, file, indent=4)

            # make a pause after 2 calls
            if i % 2 == 0:
                # get random int from 0 to 5
                pause_time = random.randint(0, 5)
                print(f"Sleeping for {pause_time} seconds")
                time.sleep(pause_time)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate claim_argument_evidence structure from requirements")
    parser.add_argument("--session_id", type=str, help="Session ID")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON or csv file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--model_id", type=str, default="gpt-4o", help="Model ID to use for generation")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use for generation")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature to use for generation")
    parser.add_argument("--max_length", type=int, default=24000, help="Maximum length of the generated text")
    parser.add_argument("--token_limit", type=int, default=8000, help="Token limit of the generated text")
    parser.add_argument("--ncalls", type=int, default=5, help="Number of calls to the model")
    parser.add_argument("--system_msg", type=str, default=None, help="System message to use for generation")
    parser.add_argument("--user_prompt", type=str, default=None, help="User prompt to use for generation")
    parser.add_argument("--prompt_template_path", type=str, default=None, help="Path to the prompt template file")
    parser.add_argument("--row_id", type=str, default="requirement_id", help="Row ID to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)
    