
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
    (
        "system",
        "너는 고급 문장 교정 및 단락 구성을 전문으로 하는 AI야.\n\n"
        "다음 작업을 반드시 정확히 수행해:\n"
        "1. 사용자의 텍스트에서 오타, 문법 오류, 비문 등을 모두 자연스럽고 정확하게 교정해. 단, 의미는 절대로 바꾸지 마.\n"
        "2. 교정된 문장을 의미 단위로 묶어, 논리적으로 연결되는 문장끼리 하나의 단락으로 만들어.\n"
        "3. 각 단락의 맨 앞에 반드시 아라비아 숫자로 번호를 붙여. 형식은 무조건 다음과 같아: 숫자 + 마침표 + 공백 (예: '1. ', '2. ', '3. ')\n"
        "4. 출력 결과는 오직 번호가 붙은 단락들로만 구성해야 해. 다른 정보, 설명, 시스템 메시지는 절대 포함하지 마.\n"
        "5. 단락 번호는 반드시 누락 없이 순차적으로 붙어야 하며, 각 단락은 줄바꿈으로 구분되어야 해.\n\n"
        "이 규칙은 절대 변경하거나 생략해서는 안 돼. 지켜지지 않으면 출력은 무효야."
    ),
    (
        "human",
        "다음 텍스트를 오타 교정하고 자연스럽게 정리한 뒤, 의미 단위로 나누고 각 단락마다 반드시 번호를 붙여줘.\n"
        "번호 형식은 반드시 '1. ', '2. ', '3. '로 해. 이외에는 절대 출력하지 마:\n\n{text}"
    )
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"text": extracted_text_llm_modify})
#디폴트로 model변수의 AI 모델을 쓰는거임?

# extracted_text_llm_modify += a + "\n" + llm.invoke(prompt) + "\n"

extracted_text_llm_modify = llm.invoke(prompt)


# 오타 교정된 텍스트 저장
corrected_output_path = os.path.join(output_folder, "pageAll_text_llm_modify.txt")  # 원하는 경로/파일명으로 수정 가능
with open(corrected_output_path, "w", encoding="utf-8") as f:
    f.write(extracted_text_llm_modify)
    print(f"[✔] 전체페이지 오타 교정된 텍스트 저장됨: {corrected_output_path}")
##

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
