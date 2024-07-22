import os
from pathlib import Path

import typer
from rich import print
from source.generator import generate_chain, llm_generator
from source.retreiver import script_retriever
from source.utils import download_file, list_files, load_env, load_yaml, upload_file
from typing_extensions import Annotated

load_env()
cli = typer.Typer(no_args_is_help=True)


@cli.command()
def file(
    mode: Annotated[str, typer.Argument(help="list, upload 또는 download")],
    src: Annotated[Path, typer.Option("--src", "-s", help="업로드 또는 다운로드할 파일 또는 폴더")] = "./data",
    dest: Annotated[Path, typer.Option("--dest", "-d", help="다운로드 폴더 경로")] = "./downloads",
):
    """파일을 리스트, 업로드 또는 다운로드합니다."""
    match mode:
        case "list":
            print(list_files())
        case "download":
            download_file(src, dest)
        case "upload":
            upload_file(src)
        case _:
            raise NotImplementedError


@cli.command()
def chat(
    config_yaml: Annotated[str, typer.Option(help="config 파일 위치")] = "config_yaml/base.yaml",
    update: Annotated[bool, typer.Option("--update")] = False,
):
    """챗봇과 대화합니다."""
    config = load_yaml(path=config_yaml)
    llm = llm_generator(generator_type=config['generator_type'], generator_model=config['generator_model'])
    db = config["db"]
    if update:
        download_file("index.faiss", os.path.join(db, "index.faiss"))
        download_file("index.pkl", os.path.join(db, "index.pkl"))

    user_input = typer.prompt("대화를 시작하세요")

    # Retriever 활용하여 script 불러오기
    script = script_retriever(config['embedding_type'], config['embedding_model'], db, query=user_input)
    print(f"검색된 script: {script}")

    # 프롬프트 생성
    prompt_template = config['prompt']
    prompt_variables = {"script": script, "question": user_input, "character": config['character']}

    # 응답 생성
    answer = generate_chain(llm, prompt_template, prompt_variables)

    print(f"AI: {answer}")


if __name__ == "__main__":
    cli()
