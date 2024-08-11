import os
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from rich import print


def load_env() -> None:
    load_dotenv()
    env_variables = [
        "OPENAI_API_KEY",
        "FILE_SERVER_URI",
        "FILE_SERVER_USERNAME",
        "FILE_SERVER_PASSWORD",
    ]
    for var in env_variables:
        if os.environ.get(var) is None:
            raise ValueError(f"Environment variable {var} is not set.")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="UTF8") as f:
        load_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return load_yaml


def list_files() -> list[str]:
    url = os.environ["FILE_SERVER_URI"] + "/files"
    auth = HTTPBasicAuth(
        os.environ["FILE_SERVER_USERNAME"],
        os.environ["FILE_SERVER_PASSWORD"],
    )
    response = requests.get(url, auth=auth)
    response.raise_for_status()
    return response.json()


def upload_file(path: Path) -> None:
    files = []
    if path.is_dir():
        for f in path.glob("**/*"):
            if f.is_file():
                files.append(f)
    elif path.is_file():
        files.append(path)
    if len(files) == 0:
        print("업로드할 파일이 없습니다.")
        return

    url = os.environ["FILE_SERVER_URI"] + "/upload"
    auth = HTTPBasicAuth(
        os.environ["FILE_SERVER_USERNAME"],
        os.environ["FILE_SERVER_PASSWORD"],
    )
    for f in files:
        print(f"파일 업로드: {f}")
        with open(f, "rb") as handle:
            response = requests.post(url, files={"file": handle}, auth=auth)
        response.raise_for_status()


def download_file(source: Path, target: Path) -> None:
    print(f"파일 다운로드: {source} -> {target}")
    url = os.environ["FILE_SERVER_URI"] + "/download/" + str(source)
    auth = HTTPBasicAuth(
        os.environ["FILE_SERVER_USERNAME"],
        os.environ["FILE_SERVER_PASSWORD"],
    )
    response = requests.get(url, auth=auth)
    response.raise_for_status()

    with open(target, "wb") as handle:
        handle.write(response.content)
