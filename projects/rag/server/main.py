from pathlib import Path

import anyio
import typer
from httpx import AsyncClient
from rich import print
from server import app
from typing_extensions import Annotated
from utils import download_file, list_files, load_env, upload_file

load_env()
cli = typer.Typer(no_args_is_help=True)


@cli.command()
def file(
    mode: Annotated[str, typer.Argument(help="list, upload 또는 download")],
    src: Annotated[
        Path, typer.Option("--src", "-s", help="업로드 또는 다운로드할 파일 또는 폴더")
    ] = "./data",
    dest: Annotated[
        Path, typer.Option("--dest", "-d", help="다운로드 폴더 경로")
    ] = "./downloads",
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
    llm: Annotated[str, typer.Option(help="언어모델 종류")] = "openai",
    retriever: Annotated[str, typer.Option(help="검색기 종류")] = "metadata",
    rag: Annotated[str, typer.Option(help="RAG 종류")] = "base",
):
    """챗봇과 대화합니다."""
    user_character = typer.prompt("캐릭터를 입력하세요")
    user_question = typer.prompt("질문을 입력하세요")

    async def _invoke():
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/invoke",
                json={
                    "llm": llm,
                    "retriever": retriever,
                    "rag": rag,
                    "character": user_character,
                    "prompt": user_question,
                },
            )
        print(response.json())

    anyio.run(_invoke)


if __name__ == "__main__":
    cli()
