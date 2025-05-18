# Harry Potter RAG Chat Bot
![Image](https://github.com/user-attachments/assets/eca8d9d8-f362-43e3-b518-969e4625d321)
> **해리포터 등장인물과 대화할 수 있는 챗봇입니다.**
> * Prompt RAG로 해리의 경험과 주변 인물을 물어보세요.

<br>

## Setting
1. `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다:
   ```bash
   cp .env.example .env
   ```

2. `.env` 파일에 아래와 같이 값을 입력합니다:

| 변수명              | 설명                                 |
|--------------------|--------------------------------------|
| `OPENAI_API_KEY`   | ✅ 필수 입력 – OpenAI API 키           |
| `RUNPOD_API_KEY`   | A/B 테스트에서 사용할 Runpod API 키 (미사용 시 공백)   |
| `SERVER_PASSWORD`  | 웹페이지 접속 시 사용할 비밀번호       |
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_API_KEY=
SERVER_PASSWORD=your_secure_password
```

## 실행 방법

1. 아래 명령어로 서버를 실행합니다:
   ```bash
   python rag/server/server.py
   ```
2. 웹 브라우저에서 다음 주소 중 하나로 접속합니다
```
http://0.0.0.0:8000
또는
http://localhost:8000
```
3. `.env` 파일의 `SERVER_PASSWORD` 비밀번호 입력합니다.


<br>

## Service Architecture

![Image](https://github.com/user-attachments/assets/9740f2c9-ff95-4c10-8444-002da5c7461f)

<br>

## RAG Architecture
### Prompt RAG

![image](https://github.com/user-attachments/assets/38de4e21-c5f8-4ed9-aaf2-a688f0b67ff9)


<br>

### Base RAG

![Image](https://github.com/user-attachments/assets/1f4b500a-572d-4771-b865-c03166c5b4cc)

<br>

### Self RAG

![Image](https://github.com/user-attachments/assets/96732015-eec3-49a6-a391-0fc13b97e747)

<br>

### Corrective RAG

![Image](https://github.com/user-attachments/assets/fb20c6f7-b9e1-4595-9b25-fc6c8ba8fab9)
