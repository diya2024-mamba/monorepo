# Run the app
# mesop app.py
import os

import mesop as me
import mesop.labs as mel
from dotenv import load_dotenv
from source.generator import generate_chain, llm_generator
from source.retreiver import script_retriever
from source.utils import load_yaml

load_dotenv()

prj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(prj_dir, 'config_yaml', 'base.yaml')
config = load_yaml(path=config_path)

llm = llm_generator(
    generator_type=config['generator_type'],
    generator_model=config['generator_model'],
)


def upper_case_stream(query: str):
    script = script_retriever(
        config['embedding_type'],
        config['embedding_model'],
        config['db'],
        query,
    )
    prompt = config['prompt']
    prompt_variables = {
        "script": script,
        "question": query,
        "character": config['character'],
    }
    print(script)
    answer = generate_chain(llm, prompt, prompt_variables)
    return answer


@me.page(
    title="Text to Text Example",
)
def app():
    mel.text_to_text(
        upper_case_stream,
        title="페르소나 챗봇",
    )
