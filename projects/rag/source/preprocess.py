import json
import fitz
import re

from utils import load_txt

from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()


def harrypotter_1_preprocess(pdf_path: str)-> None:
    '''
    data/해리포터3 대본.pdf preprocessing pdf -> txt
    '''
    eng_script = []
    kor_script = []

    eng_com = re.compile(r'\\(?[a-zA-Z]')

    doc = fitz.open(pdf_path)

    for p in range(len(doc)):
        page = doc.load_page(p)
        text = page.get_text("text").replace('\\xa0\\n', '').split('\\n')
        for t in text:
            if t == '':
                continue
            elif eng_com.match(t):
                eng_script.append(t)
            else:
                kor_script.append(t)
    doc.close()
    file_name = 'ko_script.txt'

    with open(file_name, 'w+') as file:
        file.write('\\n'.join(kor_script))
    file_name = 'eng_script.txt'

    with open(file_name, 'w+') as file:
        file.write('\\n'.join(eng_script))

    return


def preprocess_metadata(txt_path: str, save_path=None) -> list:
    '''
    input: ['character : content', ] 형식의 script.txt 파일
    
    output: list(Document(page_content=dialogue, metadata={'script_id':i, "character":character}))
    script와 metadata 저장

    save_json_path: .json으로 저장
    .json: {'script_id' :{"character":character, 'content':dialogue}}    
    '''
    script = load_txt(txt_path)
    sample = []
    sample_dict = {}
  
    for i in range(len(script)):
        if  ':' in script[i]:
            character,dialogue = script[i].split(':')
            character,dialogue = character.strip(), dialogue.strip()
            sample.append(Document(page_content=dialogue, metadata={'script_id':i, "character":character}))
            sample_dict[i] = {"character":character, 'content':dialogue}

        else:
            sample.append(Document(page_content=script[i], metadata={'script_id':i, "character":'Commentary'}))
            sample_dict[i] = {"character":'Commentary', 'content':script[i]}
    
    if save_path != None:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, indent="\t", ensure_ascii=False)

    return sample

if __name__== "__main__":
    preprocess_metadata('data/ko_script.txt', 'data/ko_script.json')