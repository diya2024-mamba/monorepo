MCQA (Multiple Choice Question Answering) 벤치마크를 평가하는데 특화된 평가툴 lm-eval-hanress입니다.
주어진 모델, 태스크, 데이터셋이 아닌 **커스텀 세팅**에서 평가하는 방법에 대해 간단히 안내합니다.

## 1. lm-eval-hanress 설치
설치하는 방법은 [링크](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)에도 잘 나와있으니 쉽게 따라할 수 있습니다.

```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

설치되는 라이브러리 및 패키지는 [여기](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/pyproject.toml)에서 확인 가능합니다.
(`lm-evaluation-harness/pyproject.toml` 파일의 링크입니다.)

대부분의 사람들이 사용하는 `transformers` 를 그대로 사용할 수 있으며, 이러한 로컬 모델의 경우 빠른 추론을 위해 `vLLM`이 이용됩니다.


## 2. 간단 추론하기

```
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```
마찬가지로 [링크](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)에 나와있는 내용입니다.

특징은 `lm_eval`에 인자를 전달하는 방식이라는 점이고, 이는 `lm-evaluation-hanress`의 root 폴더에서 실행해야 정상적으로 동작합니다.
만약 해당 경로가 제대로 인식되지 않는 경우 환경 변수에 등록해줘야 합니다.
`export PYTHONPATH=$PYTHONPATH:/example/lm-evaluation-harness`

제대로 등록되었는지 확인하는 방법으로는 `lm-eval --task list` 명령어를 사용하는 것이 있습니다.
만약 이를 실행했을 때 터미널에 텅빈 리스트가 출력된다면 경로가 제대로 등록되지 않은 것입니다.

## 3. 인자 알아보기
[인터페이스 관련 문서](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)를 보시면 CLI 환경에서 쓸 수 있는 인자에 대해 자세히 설명하고 있습니다.
대표적인 것 몇 개만 설명하면 이렇습니다.

- `--model`: 허깅페이스의 모델을 쓸 것이라면 `hf`를 입력하면 됩니다. 이외에 `openai-completions`, `anthropic` 등 다양한 api를 사용할수도 있습니다. 전체 목록은 [여기](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#model-apis-and-inference-servers)에서 볼 수 있습니다.
- `--model_args`: 모델을 로드할 때 전달하는 인자를 여기에서 전달합니다. 예를 들어 `model_name_or_path`에 관한 것을 전달할수도 있고, `cache_dir` 또는 `dtype`을 지정할수도 있습니다.
- `--task`: 어떤 벤치마크를 돌릴지 결정합니다. `--task lambada_openai,arc_easy`와 같이 쉼표로 구분하여 여러 개의 벤치마크를 한 번에 돌릴수도 있습니다.
- `--log_samples`: 결과를 저장할 때 sampling 결과를 포함할지 결정합니다. 이를 사용하지 않으면 각 태스크별 점수 파일만 생성됩니다.
- `--batch_size`: 배치 사이즈를 결정합니다. `auto:4`와 같이 설정하면 알아서 최대 배치사이즈를 계산해서 찾아줍니다.

이외에도 조정 가능한 옵션들이 굉장히 많으니 [문서](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)를 꼭 참고해보시기 바랍니다!


## 4. 커스텀 데이터셋
벤치마크가 여러 개 세팅되어 있긴 하지만 상황에 따라 다르게 테스트 할 수 있습니다.
예를 들어 few-shot을 바꿔줄 수도 있고, few-shot의 개수 자체를 조정할 수도 있겠죠.
또는 벤치마크에 포함된 모든 데이터셋에 대한 추론이 불필요할 수도 있습니다.

물론 추론 속도가 빠른 편이라 그냥 추론 결과 중 선별하는 게 맘이 편할 수도 있습니다만 이를 컨트롤하는 방법을 알아두는 게 맘이 편하겠더라고요.
그래서 여기서는 `MMLU` 벤치마크의 일부 태스크만 선정해서 돌릴 수 있도록 세팅하는 방법에 대해 간략히 소개드리겠습니다.

- `lm_eval/tasks/mmlu/` 경로에 `custom` 폴더를 추가해줍니다.
- 기존에는 `default`, `continuation`, `flan_cot_fewshot` 등의 폴더가 존재합니다. 이것들과 같은 구성으로 폴더를 .yaml 파일로 채우면 됩니다.
- `mmlu` 벤치마크의 경우 크게 세 종류의 파일이 필요합니다.
    1. `_mmlu.yaml`: 어떤 종류의 하위 그룹을 취할지 결정하는 내용이 포함됩니다. task 하위에 group을 여러 개 지정하면 됩니다. 이때 하위 그룹의 이름은 group 아래 task라고 추가 지정하게 되어 있습니다. 예시 파일에는 제가 stem만 지정했는데, 꼭 stem이 아니어도 되고 humanities와 같은 다른 그룹도 포함 가능합니다.
    2. `mmlu_computer_security.yaml`:  _mmlu.yaml 파일에서 하위 그룹으로 지정한 태스크 중 하나입니다. 여기서는 tag를 통해 어떤 하위 그룹에 속하는지를 밝히고, 이 태스크 이름 자체는 task에 표시합니다. 그리고 어떤 것으로 평가를 수행할지는 include에 명시할 수 있습니다.
    3. `_mmlu_custom_template_yaml`: 파일 이름에 주의합니다. `.yaml`이 아니라 `_yaml`입니다. 여기서는 어떤 데이터셋을 불러올지, few-shot을 쓸지, output 형식은 어떻게 할지, 데이터를 불러올 때 적용할 함수는 무엇인지 정할 수 있습니다.
- 예시에 해당하는 'custom 폴더'를 참고해보시면 도움이 될 것입니다.

## 5. 커스텀 함수
위에서 사용한 예시인 custom 폴더에서 작업을 한다고 하면, 해당 폴더 내에 `utils.py` 파일을 만들 수 있습니다.
여기에서 doc_to_text, doc_to_choice, doc_to_target에 들어갈 함수를 덮어 씌울 수 있습니다.

이 함수들은 `jinja2`로 직접 작성할수도 있고, 방금 말씀드린 것처럼 `utils.py`에 작성 후 `!function utils.shuffle_and_format_choices`와 같이 불러올수도 있습니다.
[jinja 관련 설명](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#writing-a-prompt-with-jinja-2) 또는 [python function 관련 설명](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#using-python-functions-for-prompts)을 참고해보세요!

저의 경우에는 최종적으로 `_mmlu_custom_template_yaml` 파일에

```yaml
doc_to_text: !function utils.shuffle_and_format_choices
doc_to_choice: !function utils.get_shuffled_choice_labels
doc_to_target: !function utils.get_shuffled_target
```

이와 같이 함수를 불러오도록 세팅했습니다.

## 6. 커스텀 MMLU 실행하기
저는 설정을 한 번에 편하게 맞춰두고 돌리기 위해서 스크립트를 하나 작성했습니다.
경로만 일부 감춘 파일 전문은 다음과 같습니다.

```bash
#!/bin/bash

# .env 파일에서 환경 변수 내보내기
set -a
source /private_path/.env
set +a

# lm-evaluation-harness 폴더로 디렉토리 변경
cd /private_path/lm-evaluation-harness

MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
CACHE_DIR=/private_path/.cache/huggingface
OUTPUT_PATH=/private_path/custom_mmlu/1

# 커스텀 MMLU 태스크에 대한 평가 실행
lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,dtype=bfloat16,cache_dir=$CACHE_DIR \
    --tasks mmlu_custom \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path $OUTPUT_PATH \
    --log_samples
```

간단한 설명을 드리면 다음과 같습니다.

- `.env` 파일에는 허깅페이스 토큰이 들어있습니다. llama 모델을 사용할 때 인증이 필요하기 때문입니다. 참고로 이 토큰의 이름은 `HUGGING_FACE_HUB_TOKEN`로 저장되어야 합니다.
    - 이때 `set -a`, `set +a` 를 사용하면 해당 파일을 `.env` 파일을 이용해 환경변수를 안전하게 등록할 수 있습니다.
- 모델은 허깅페이스의 llama-3-8b-instruct를 사용했고, `bfloat16` 자료형으로 불러왔습니다.
- 또한 위에서 말씀드렸던 것처럼 batch_size는 알아서 최대치를 사용할 수 있도록 `auto`로 지정했습니다.
