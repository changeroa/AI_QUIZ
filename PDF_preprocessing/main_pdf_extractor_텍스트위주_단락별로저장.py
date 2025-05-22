
import fitz  # PyMuPDF
import os
from PIL import Image
import pytesseract
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

llm2 = OllamaLLM(model="gemma3:12b")
llm = OllamaLLM(model="gemma3:4b")

# Tesseract 언어 설정 (한글 + 영어)
tesseract_lang = "kor+eng"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# 추출할 파일
doc = fitz.open(r"d:\고홍규\2025 한국외대 수학과 2학년 2학기\Quiz생성ai 프로젝트\플라톤향연.pdf")



# 현재 시간으로 이미지저장폴더명 생성
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(
    r"d:\고홍규\2025 한국외대 수학과 2학년 2학기\Quiz생성ai 프로젝트", f"추출_{now_str}"
)
os.makedirs(output_folder, exist_ok=True)


# =================
# 오타교정 (방법3가지.)

extracted_text_llm_modify =''
# 텍스트 추출
for page in doc:
    extracted_text_llm_modify +=  page.get_text() 
print("[텍스트]", extracted_text_llm_modify)
# text = pdf에서 순수 추출된 단어정보

# ===============================
# 오타교정 (방법3가지.)



# PART 3: Prompt with System and Human Messages (Using Tuples)
messages = [
            ("system", "너는 고급 문장 교정 및 청킹을 전문으로 하는 AI야. 사용자가 입력한 텍스트에 대해 다음 작업을 수행해.\n\n"
"1. 먼저, 문법, 철자, 오타를 모두 자연스럽고 정확하게 교정해. 의미를 보존하되, 가능한 한 자연스러운 현대 한국어 표현으로 바꿔.\n"
"2. 교정된 문장들을 의미 단위로 묶어서 단락을 나눠줘. 이때 연관된 주제끼리 한 단락에 들어가도록 하고, 단락마다 번호를 붙여: 예) '1', '2', '3'...\n"
"3. 출력은 반드시 교정된 문장과 번호가 매겨진 단락으로 구성된 최종 텍스트만 보여줘. 그 외의 설명, 시스템 메시지, 메타 정보는 절대 출력하지 마.\n"
"단, 오타 교정이 끝난 문장은 재배열하거나 단락으로 묶을 수는 있지만 내용이나 표현을 다시 수정하거나 바꿔서는 안 돼."
),

            ("human", 
"다음 텍스트의 오타를 교정하고 문장을 자연스럽게 정리해줘. 그 다음, 연관된 내용끼리 단락을 나눠서 묶고 단락마다 숫자를 붙여. 단락마다 반드시 번호를 붙여 예) '1', '2', '3'...\n 그 외에는 아무것도 출력하지 마. :  \n\n{text}"),

        ]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"text": extracted_text_llm_modify})
#디폴트로 model변수의 AI 모델을 쓰는거임?

# extracted_text_llm_modify += a + "\n" + llm.invoke(prompt) + "\n"

extracted_text_llm_modify = llm.invoke(prompt)


import re

# 1. 단락별로 분리
# "숫자." 또는 "숫자 " 형태로 단락이 시작된다고 가정
pattern = re.compile(r"(\d+)\.\s*(.*?)((?=\n\d+\. )|$)", re.DOTALL)

matches = pattern.findall(extracted_text_llm_modify)

# matches는 [(번호, 단락내용, 다음단락시작전까지 텍스트), ...] 튜플 리스트

for num, content, _ in matches:
    filename = os.path.join(output_folder, f"paragraph_{num}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        # content는 단락 내용, 혹시 앞뒤 공백 제거
        f.write(content.strip())
    print(f"[✔] 단락 {num} 저장됨: {filename}")
