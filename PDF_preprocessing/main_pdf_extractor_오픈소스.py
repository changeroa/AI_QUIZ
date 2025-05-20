from vision_parse import VisionParser
from langchain_ollama import OllamaLLM


llm2 = OllamaLLM(model="gemma3:12b")
llm = OllamaLLM(model="gemma3:4b")

custom_prompt = """
Strictly preserve markdown formatting during text extraction from scanned document.
"""

# Initialize parser with Ollama configuration
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.6,
    num_ctx=4096,
    image_mode="base64",
    custom_prompt=custom_prompt,
    detailed_extraction=True,
    ollama_config={
        "OLLAMA_NUM_PARALLEL": 8,
        "OLLAMA_REQUEST_TIMEOUT": 240,
    },
    enable_concurrency=True,
)

# Convert PDF to markdown
pdf_path = "OSS9.pdf" # local path to your pdf file
markdown_pages = parser.convert_pdf(pdf_path)

for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")
