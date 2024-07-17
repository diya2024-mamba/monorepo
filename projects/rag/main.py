# Run the app
# mesop main.py
import os

import mesop as me
import mesop.labs as mel

from utils import load_yaml
from generator import llm_generator, generate_chain
from retreiver import script_retriever


prj_dir  = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(prj_dir, 'config_yaml', 'base.yaml')
config = load_yaml(path=config_path)


llm = llm_generator(generator_type=config['generator_type'], generator_model=config['generator_model'])


@me.page(
  path="/text_to_text",
  title="Text to Text Example",
)
def app():
  mel.text_to_text(
    upper_case_stream,
    title="페르소나 챗봇",
  )


def upper_case_stream(query: str):
  script = script_retriever(config['embedding_type'], 
                            config['embedding_model'], 
                            config['db'],
                            query)

  prompt = config['prompt']
  prompt_variables = {"script": script,
                      "question": query,
                      "character": config['character']}
  print(script)
  answer = generate_chain(llm, prompt, prompt_variables)
  return answer