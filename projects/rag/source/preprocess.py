import json
import re
import os

import fitz
from dotenv import load_dotenv
from langchain.docstore.document import Document
from utils import load_txt

load_dotenv()


def harrypotter_3_preprocess(pdf_path: str) -> None:
    """
    data/해리포터3 대본.pdf preprocessing pdf -> txt
    """
    eng_script = []
    kor_script = []

    eng_com = re.compile(r"\(?[a-zA-Z]")

    doc = fitz.open(pdf_path)

    for p in range(len(doc)):
        page = doc.load_page(p)
        text = page.get_text("text").replace("\xa0\n", "").split("\n")
        for t in text:
            if t == "":
                continue
            elif eng_com.match(t):
                eng_script.append(t)
            else:
                kor_script.append(t)
    doc.close()

    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    path  = os.path.join(path, 'data')

    file_name = os.path.join(path, "ko_script3.txt")
    with open(file_name, "w+", encoding='utf-8') as file:
        file.write("\n".join(kor_script))

    file_name = os.path.join(path, "eng_script3.txt")
    with open(file_name, "w+", encoding='utf-8') as file:
        file.write("\n".join(eng_script))

    return

def preprocess_txt(txt_path, save_name):
    f = open(txt_path,  encoding='utf-8')
    lines = f.readlines()
    script = []
    before_line = ''
    for line in lines:
        if line.find(':') > -1 or line.find('(') > -1 :
            script.append(before_line.replace('\n',''))
            before_line = line
        else:
            before_line = before_line + line
    
    with open(save_name, 'w+', encoding='utf-8') as file:
        file.write('\n'.join(script))

def preprocess_metadata(txt_path: str, save_path=None) -> list:
    """
    input: ['character : content', ] 형식의 script.txt 파일

    output: list(Document(page_content=dialogue, metadata={'script_id':i, "character":character}))
    script와 metadata 저장

    save_json_path: .json으로 저장
    .json: {'script_id' :{"character":character, 'content':dialogue}}
    """
    script = load_txt(txt_path)
    sample = []
    sample_dict = {}

    for i in range(len(script)):
        if ":" in script[i]:
            d = script[i].split(":")
            character, dialogue = d[0], d[1]
            character, dialogue = character.strip(), dialogue.strip()
            sample.append(
                Document(
                    page_content=dialogue,
                    metadata={"script_id": i, "character": character},
                )
            )
            sample_dict[i] = {"character": character, "content": dialogue}

        else:
            sample.append(
                Document(
                    page_content=script[i],
                    metadata={"script_id": i, "character": "Commentary"},
                )
            )
            sample_dict[i] = {"character": "Commentary", "content": script[i]}

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sample_dict, f, indent="\t", ensure_ascii=False)

    return sample


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    path  = os.path.join(path, 'data')
    
    harrypotter_3_preprocess(os.path.join(path, '해리포터3 대본.pdf'))
    
    txt_path = os.path.join(path, 'ko_script3.txt')
    preprocess_txt(txt_path, txt_path)
    preprocess_metadata(txt_path, os.path.join(path, 'ko_script3.json'))