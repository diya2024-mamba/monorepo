import argparse
import json
import os
import random
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import anthropic
import google.generativeai as genai
import openai
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class CustomDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item


def collate_fn(batch):

    datas = [x for x in batch]

    return datas


def get_client(args):
    API_KEY = "Wrtie your API key in .env file"
    # OpenAI API key만 지원 (240810)
    if args.model_name in ["gpt-4", "gpt-4o", "gpt-4o-mini"]:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
        )
    elif args.model_name in ["deepseek-chat", "deepseek-coder"]:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/")
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]:
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        client = anthropic.Anthropic(
            api_key=API_KEY,
        )
    else:
        client = None
        print(
            "For other model API calls, please implement the client definition method yourself."
        )
    return client


def call_api(args, client, instruction, inputs):
    if args.model_name in [
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "deepseek-chat",
        "deepseek-coder",
    ]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            temperature=0,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]:
        chat_session = client.start_chat(history=[])
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system="",
            messages=[{"role": "user", "content": instruction + inputs}],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    else:
        print(
            "For other model API calls, please implement the request method yourself."
        )
        result = None
    return result


def load_mmlu_pro(shuffle: str = "no") -> Tuple[Dict, Dict]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df, shuffle)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(df: Dataset, shuffle: str = "no") -> Dict[str, List]:
    res_df = []
    for each in df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []

        if shuffle == "no":
            pass
        elif shuffle == "reverse":
            each["options"] = each["options"][::-1]
            choice_map = "ABCDEFGHIJ"[: len(each["options"])][::-1]

            each["answer_index"] = choice_map.index(each["answer"])
            each["answer"] = "ABCDEFGHIJ"[each["answer_index"]]
        else:  # shuffle == "random"
            random.seed(42)
            indices = [i for i in range(len(each["options"]))]
            random.shuffle(indices)

            each["options"] = [each["options"][i] for i in indices]
            each["answer_index"] = indices.index(each["answer_index"])
            each["answer"] = "ABCDEFGHIJ"[each["answer_index"]]

        res[each["category"]].append(each)

    return res


def make_table(sentences_lists, choice_map="ABCDEFGHIJ") -> str:

    wrapped_sentences = [
        textwrap.wrap(sentences, width=100) for sentences in sentences_lists
    ]
    max_len = max(len(sentences) for sentences in wrapped_sentences)

    # 글자 수 차이가 날 때 공백 메우기
    for i in range(len(wrapped_sentences)):
        wrapped_sentences[i] += [""] * (max_len - len(wrapped_sentences[i]))

    headers = (
        "| " + " | ".join(choice_map[i] for i in range(len(wrapped_sentences))) + " |\n"
    )
    separators = "| " + " | ".join("-" for _ in range(len(wrapped_sentences))) + " |\n"

    markdown_table = headers + separators
    for row in zip(*wrapped_sentences):
        markdown_table += "| " + " | ".join(row) + " |\n"

    return markdown_table


def format_example(args, question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    if args.table:
        example += make_table(options, choice_map)
    else:
        for i, opt in enumerate(options):
            example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request_dict(args, client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if (
            q_id == each["question_id"]
            and single_question["question"] == each["question"]
        ):
            pred = extract_answer(each["model_outputs"])
            return {"pred": pred, "response": each["model_outputs"], "exist": exist}
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]

    prompt = (
        "The following are multiple choice questions (with answers) about {}. Think step by"
        ' step and then output the answer in the format of "The answer is (X)" at the end.\n\n'.format(
            category
        )
    )
    for each in cot_examples:
        prompt += format_example(
            args, each["question"], each["options"], each["cot_content"]
        )
    input_text = format_example(args, question, options)
    try:
        # start = time.time()
        response = call_api(args, client, prompt, input_text)
        # print("requesting gpt 4 costs: ", time.time() - start)
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return {"pred": pred, "response": response, "exist": exist}


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            random.seed(12345)
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                                # print("random hit.")
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if (
            single["question_id"] == curr["question_id"]
            and single["question"] == curr["question"]
        ):
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate_batch(args, subjects):
    client = get_client(args)
    test_df, dev_df = load_mmlu_pro(shuffle=args.shuffle)
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        dataloader = iter(
            DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        )

        for batch in tqdm(dataloader):
            # label = datas["answer"]
            # category = subject

            with ThreadPoolExecutor() as executor:
                dict_list = list(
                    executor.map(
                        lambda x: single_request_dict(args, client, x, dev_df, res),
                        batch,
                    )
                )

            category = subject
            for result, data in zip(dict_list, batch):
                label = data["answer"]
                response, pred = result["response"], result["pred"]
                if response is not None:
                    res, category_record = update_result(output_res_path)
                    if category not in category_record:
                        category_record[category] = {"corr": 0.0, "wrong": 0.0}
                    data["pred"] = pred
                    data["model_outputs"] = response
                    merge_result(res, data)
                    if pred is not None:
                        if pred == label:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                    save_res(res, output_res_path)
                    save_summary(category_record, output_summary_path, args.shuffle)
                    res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path, args.shuffle)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path, shuffle=None):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}

    if shuffle is not None:
        category_record["shuffle"] = shuffle

    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./data/eval_results_api"
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="gpt-4o-mini",
        choices=[
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "deepseek-chat",
            "deepseek-coder",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
        ],
    )
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--shuffle", type=str, choices=["reverse", "random", "no"], default="no"
    )
    parser.add_argument("--table", type=bool, choices=[True, False], default=False)
    args = parser.parse_args()

    load_dotenv()

    assigned_subjects = []
    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_batch(args, assigned_subjects)
