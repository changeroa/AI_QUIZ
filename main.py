"""
PDF 텍스트 추출기 - text_analyzer 임계값 기반 페이지별 분류 (개선된 버전)

text_analyzer로 계산된 임계값을 활용하여 PDF 텍스트를 페이지별로 분류하고
단일 JSON 파일로 출력하는 시스템
"""

import fitz
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """PDF 텍스트를 임계값 기반으로 페이지별 분류하여 추출하는 클래스"""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        초기화
        
        Args:
            thresholds: text_analyzer에서 계산된 임계값
                       {"title_threshold": 23.0, "subtitle_threshold": 20.88, 
                        "body_size": 15.0, "small_text_threshold": 12.0}
        """
        if thresholds is None:
            # 기본 임계값 (실제 테스트 결과 기반 최적화)
            self.thresholds = {
                "title_threshold": 30.0,   # 제목: 30pt 이상 (실제 테스트 최적값)
                "subtitle_threshold": 24.0, # 소제목: 24pt 이상 (실제 테스트 최적값)
                "body_size": 20.0,         # 본문: 20pt 이상 (실제 테스트 최적값)
                "small_text_threshold": 12.0 # 작은 텍스트: 12pt 이상 (실제 테스트 최적값)
            }
        else:
            self.thresholds = thresholds
        
        logger.info(f"분류 임계값: {self.thresholds}")
    
    def classify_text_element(self, size: float, text: str, bbox: List[float], 
                             page_width: float, page_height: float, 
                             font: str = "", flags: int = 0) -> str:
        """
        텍스트 요소를 크기와 특성 기반으로 분류
        
        Args:
            size: 폰트 크기
            text: 텍스트 내용
            bbox: 경계 박스 [x0, y0, x1, y1]
            page_width: 페이지 너비
            page_height: 페이지 높이
            font: 폰트명
            flags: 폰트 플래그 (볼드, 이탤릭 등)
            
        Returns:
            분류된 카테고리: 'Titles', 'Subtitles', 'Body Text', 'Others'
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()
        x0, y0, x1, y1 = bbox
        
        # 1. 위치 기반 우선 분류 (완화된 조건)
        is_landscape = page_width > page_height
        
        if is_landscape:  # 가로형 슬라이드
            # 상단 2% 영역으로 줄임
            if y1 > page_height * 0.9:
                return "Others"
            
            # 하단 3% 영역으로 줄임
            if y0 < page_height * 0.05:
                return "Others"
            
            # 우측 하단 모서리 (페이지 번호 위치) - 조건 완화
            if (x0 > page_width * 0.9 and y0 < page_height * 0.1):
                if len(text_clean) <= 4:  # 길이 조건 완화
                    return "Others"
        else:  # 세로형 문서
            if y1 > page_height * 0.98 or y0 < page_height * 0.02:  # 조건 완화
                return "Others"
        
        # 2. 패턴 기반 분류 (더 엄격하게)
        # 페이지 번호 패턴 - 위치도 고려
        page_number_patterns = [
            text_clean.isdigit() and len(text_clean) <= 3,
            text_clean in ['i', 'ii', 'iii', 'iv', 'v'],
            '/' in text_clean and len(text_clean.split('/')) == 2 and all(part.isdigit() for part in text_clean.split('/') if part),
            text_clean.lower().startswith('page') and len(text_clean) <= 10,
            text_clean.startswith('p.') and len(text_clean) <= 6,
            text_clean.startswith('#') and len(text_clean) <= 4
        ]
        
        # 페이지 번호로 판단되는 경우 + 위치 조건
        if any(page_number_patterns):
            # 하단이나 우측 하단에 위치한 경우만 페이지 번호로 분류
            is_bottom = y0 < page_height * 0.15
            is_right_bottom = (x0 > page_width * 0.8 and y0 < page_height * 0.2)
            if is_bottom or is_right_bottom:
                return "Others"
        
        # 수식, 참조 번호 등
        if len(text_clean) <= 8 and any([
            (text_clean.startswith('(') and text_clean.endswith(')')),
            (text_clean.startswith('[') and text_clean.endswith(']')),
            text_clean.startswith('Eq.'),
            text_clean.startswith('식'),
            text_clean.startswith('Fig.'),
            text_clean.startswith('그림')
        ]):
            return "Others"
        
        # 3. 크기 기반 분류 (text_analyzer 임계값 활용)
        if size >= self.thresholds["title_threshold"]:
            return "Titles"
        elif size >= self.thresholds["subtitle_threshold"]:
            return "Subtitles"
        elif size >= self.thresholds["small_text_threshold"]:
            return "Body Text"
        else:
            return "Others"
    
    def extract_page_texts(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """
        단일 페이지에서 텍스트를 추출하고 분류
        
        Args:
            page: PyMuPDF 페이지 객체
            page_num: 페이지 번호
            
        Returns:
            페이지별 분류된 텍스트 데이터
        """
        text_dict = page.get_text("dict")
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # 분류별 텍스트 저장
        classified_texts = {
            "Titles": [],
            "Subtitles": [],
            "Body Text": [],
            "Others": []
        }
        
        # 전체 텍스트 요소 저장 (상세 정보 포함)
        all_elements = []
        
        for block_idx, block in enumerate(text_dict["blocks"]):
            if "lines" not in block:  # 텍스트가 아닌 블록은 스킵
                continue
                
            for line_idx, line in enumerate(block["lines"]):
                for span_idx, span in enumerate(line["spans"]):
                    text = span["text"].strip()
                    if not text:  # 빈 텍스트는 무시
                        continue
                    
                    size = round(span["size"], 1)
                    if size <= 0:  # 비정상적인 크기 무시
                        continue
                    
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    font = span.get("font", "")
                    flags = span.get("flags", 0)
                    
                    # 텍스트 분류
                    category = self.classify_text_element(
                        size, text, bbox, page_width, page_height, font, flags
                    )
                    
                    # 분류별로 텍스트 저장
                    classified_texts[category].append(text)
                    
                    # 상세 정보 저장
                    element = {
                        "text": text,
                        "category": category,
                        "size": size,
                        "font": font,
                        "bbox": [round(x, 2) for x in bbox],
                        "flags": flags,
                        "block_idx": block_idx,
                        "line_idx": line_idx,
                        "span_idx": span_idx
                    }
                    all_elements.append(element)
        
        # 페이지 데이터 구성
        page_data = {
            "page": page_num,
            "width": round(page_width, 2),
            "height": round(page_height, 2),
            "is_landscape": page_width > page_height,
            "classified_texts": classified_texts,
            "text_counts": {
                "Titles": len(classified_texts["Titles"]),
                "Subtitles": len(classified_texts["Subtitles"]),
                "Body Text": len(classified_texts["Body Text"]),
                "Others": len(classified_texts["Others"]),
                "total": len(all_elements)
            },
            "elements": all_elements  # 상세 정보
        }
        
        return page_data
    
    def extract_pdf_to_json(self, pdf_path: str, output_path: Optional[str] = None, 
                           include_details: bool = True) -> Dict[str, Any]:
        """
        PDF 전체를 페이지별로 분류하여 JSON으로 추출
        
        Args:
            pdf_path: PDF 파일 경로
            output_path: 출력 JSON 파일 경로 (None이면 파일로 저장하지 않음)
            include_details: 상세 요소 정보 포함 여부
            
        Returns:
            페이지별 분류된 전체 데이터
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        logger.info(f"PDF 텍스트 추출 시작: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # 전체 문서 데이터
            document_data = {
                "pdf_name": pdf_path.stem,
                "pdf_path": str(pdf_path),
                "total_pages": len(doc),
                "thresholds_used": self.thresholds,
                "pages": []
            }
            
            # 전체 통계
            total_stats = {
                "Titles": 0,
                "Subtitles": 0,
                "Body Text": 0,
                "Others": 0,
                "total_elements": 0
            }
            
            # 각 페이지 처리
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self.extract_page_texts(page, page_num + 1)
                
                # 상세 정보 제외 옵션
                if not include_details:
                    page_data.pop("elements", None)
                
                document_data["pages"].append(page_data)
                
                # 전체 통계 업데이트
                for category, count in page_data["text_counts"].items():
                    if category in total_stats:
                        total_stats[category] += count
                    if category == "total":
                        total_stats["total_elements"] += count
                
                logger.info(f"페이지 {page_num + 1} 처리 완료: "
                           f"제목 {page_data['text_counts']['Titles']}개, "
                           f"소제목 {page_data['text_counts']['Subtitles']}개, "
                           f"본문 {page_data['text_counts']['Body Text']}개")
            
            doc.close()
            
            # 전체 통계 추가
            document_data["total_statistics"] = total_stats
            
            # JSON 파일로 저장
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(document_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"결과 저장 완료: {output_path}")
            
            logger.info(f"추출 완료! 총 {len(document_data['pages'])}페이지, "
                       f"전체 요소 {total_stats['total_elements']}개")
            
            return document_data
            
        except Exception as e:
            logger.error(f"추출 중 오류 발생: {e}")
            raise

def load_thresholds_from_analysis(analysis_json_path: str) -> Dict[str, float]:
    """
    text_analyzer의 analysis_results.json에서 임계값 로드
    
    Args:
        analysis_json_path: analysis_results.json 파일 경로
        
    Returns:
        임계값 딕셔너리
    """
    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        thresholds = analysis_data.get("thresholds", {})
        logger.info(f"임계값 로드 완료: {thresholds}")
        return thresholds
        
    except Exception as e:
        logger.error(f"임계값 로드 실패: {e}")
        raise

def extract_pdf_with_analysis_thresholds(pdf_path: str, analysis_json_path: str, 
                                       output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    text_analyzer 결과를 활용한 PDF 텍스트 추출 (편의 함수)
    
    Args:
        pdf_path: PDF 파일 경로
        analysis_json_path: text_analyzer의 analysis_results.json 경로
        output_path: 출력 JSON 파일 경로
        
    Returns:
        추출된 데이터
    """
    # 임계값 로드
    thresholds = load_thresholds_from_analysis(analysis_json_path)
    
    # 추출기 초기화 및 실행
    extractor = PDFTextExtractor(thresholds)
    return extractor.extract_pdf_to_json(pdf_path, output_path)

# 사용 예시
if __name__ == "__main__":
    try:
        # 방법 1: analysis_results.json을 활용한 추출
        result = extract_pdf_with_analysis_thresholds(
            pdf_path="sample.pdf",
            analysis_json_path="./analysis_output/analysis_results.json",
            output_path="./output/extracted_texts.json"
        )
        
        # 결과 요약 출력
        print("\n=== 추출 결과 요약 ===")
        print(f"PDF: {result['pdf_name']}")
        print(f"총 페이지: {result['total_pages']}")
        print(f"사용된 임계값: {result['thresholds_used']}")
        print(f"전체 통계: {result['total_statistics']}")
        
        # 페이지별 요약
        print(f"\n=== 페이지별 요약 ===")
        for page_data in result['pages'][:5]:  # 처음 5페이지만 출력
            print(f"페이지 {page_data['page']}: "
                  f"제목 {page_data['text_counts']['Titles']}개, "
                  f"소제목 {page_data['text_counts']['Subtitles']}개, "
                  f"본문 {page_data['text_counts']['Body Text']}개")
        
        if len(result['pages']) > 5:
            print(f"... (총 {len(result['pages'])}페이지)")
        
        # 방법 2: 직접 임계값 지정
        # extractor = PDFTextExtractor({
        #     "title_threshold": 23.0,
        #     "subtitle_threshold": 20.88,
        #     "body_size": 15.0,
        #     "small_text_threshold": 12.0
        # })
        # result = extractor.extract_pdf_to_json("sample.pdf", "output.json")
        
    except Exception as e:
        print(f"오류 발생: {e}")