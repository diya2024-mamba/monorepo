import os

from source.utils import load_yaml, get_argumnets
from source.generator import llm_generator, generate_chain
from source.retreiver import script_retriever

from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    args = get_argumnets()
    config = load_yaml(path=args.config_yaml)

    llm = llm_generator(
        generator_type=config['generator_type'], 
        generator_model=config['generator_model']
    )

    user_input = input("대화를 시작하세요: ")

    # Retriever 활용하여 script 불러오기
    script = script_retriever(
        config['embedding_type'],
        config['embedding_model'],
        config['db'],
        query=user_input
    )
    print(f"검색된 script: {script}")

    # 프롬프트 생성
    prompt_template = config['prompt']
    prompt_variables = {"script": script,
                        "question": user_input,
                        "character": config['character']}
    

    # 응답 생성
    answer = generate_chain(llm, prompt_template, prompt_variables)

    print(f"AI: {answer}")



