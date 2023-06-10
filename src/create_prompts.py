import os
import sys
import json
import argparse
from prompt_templates import templates

def load_data_file(fpath):
    with open(fpath, 'r') as fp:
        data = json.load(fp)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Data file JSON")
    parser.add_argument("--expt_name", help="Key for which prompt template to use")
    parser.add_argument("--save_path", help="Path to save prompt inputs")
    parser.add_argument("--num_msg", type=int, default=20, help="Number of messages to use")
    args, _ = parser.parse_known_args()
    return args

def aggregate_messages_from_user(user_data, num_msgs, ordering):
    msg_data = user_data['msg_data']
    if ordering == 'most_recent_last':
        msg_data.reverse()
        msg_data = msg_data[len(msg_data)-num_msgs:]
    msg_str = ""
    for i in range(num_msgs):
        msg_str = f"{msg_str}{msg_data[i]['message']}\n"
    return msg_str

def main(args):
    data = load_data_file(args.data_path)
    user_msg_data_for_prompts = {}
    for user_id, user_data in data.items():
        aggr_msg = aggregate_messages_from_user(user_data, args.num_msg, 'most_recent_last')
        user_msg_data_for_prompts[user_id] = {}
        user_msg_data_for_prompts[user_id]["raw_text"] = aggr_msg
        user_msg_data_for_prompts[user_id]["prompt"] = templates[args.expt_name].format(aggr_msg)
    with open(args.save_path, 'w+') as fp:
        json.dump(user_msg_data_for_prompts, fp, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)
