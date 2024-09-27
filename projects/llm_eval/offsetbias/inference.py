import argparse
import json

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from utils import CustomDatasetForDev
from vllm_inference import CausalLMWithvLLM


def main(args):
    prompt_config = OmegaConf.load(args.prompt_path)
    prompt_template = prompt_config['base_prompt']
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ds = CustomDatasetForDev(args, tokenizer, prompt_template)
    llm = CausalLMWithvLLM(
        model_path=args.model_path,
        use_chat_template=False,
        verbose=False,
        model_kwargs={
            "max_model_len": 4096,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.9,
        },
        generation_config={
            "temperature": 0,
            "max_tokens": 4096,
            "repetition_penalty": 1.0,
        },
    )
    outputs = []
    preds = llm(ds.inp)
    outputs.extend(preds)
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(outputs, ensure_ascii=False, indent=4))
        print(f"추론 결과가 {args.output_file}에 저장되었습니다.")
    except IOError as e:
        raise IOError(f"{args.output_file} 파일을 저장하는 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for summarization model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True, default="/projects/llm_eval/offsetbias/prompt.yaml", help="Set prompt yaml file path",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file."
    )
    parser.add_argument(
        "--reverse", type=bool, required=False, default=False, help="Set Swap arg for testing",
    )
    args = parser.parse_args()

    main(args)
