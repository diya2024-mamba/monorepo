# LLM Generator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def huggingface_pipeline(
    generator_model: AutoModelForCausalLM, tokenizer: AutoTokenizer
):
    """huggingface model을 활용하여 llm 연동"""
    pipe = pipeline(
        "text-generation",
        model=generator_model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        top_k=50,
        temperature=0.1,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        return_full_text=False,
    )
    huggingface_llm = HuggingFacePipeline(pipeline=pipe)
    return huggingface_llm


def chat_open_ai_llm(generator_model="gpt-3.5-turbo"):
    """OpenAI API model을 활용하여 llm 연동"""
    open_ai_llm = ChatOpenAI(
        temperature=0.1, max_tokens=2048, model_name=generator_model
    )
    return open_ai_llm


def llm_generator(generator_type, generator_model):
    """LLM을 불러온다."""
    if generator_type == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(generator_model)
        model = AutoModelForCausalLM.from_pretrained(generator_model, load_in_4bit=True)
        llm = huggingface_pipeline(model, tokenizer)
    else:
        llm = chat_open_ai_llm(generator_model)

    return llm


def generate_chain(llm, prompt, prompt_variables):
    """LLM을 프롬프트 템플릿과 연동하여 응답을 생성한다."""
    prompt_template = PromptTemplate.from_template(prompt)
    output_parser = StrOutputParser()

    chain = prompt_template | llm | output_parser
    answer = chain.invoke(prompt_variables)

    return answer
