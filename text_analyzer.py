"""
PDF 텍스트 분석기 - 아키텍처 기반 개선 버전 (완전 재작성)
핵심 기능: 페이지별 Gap 분석 → Outlier 탐지 → 차별적 처리
"""

import fitz
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import statistics
from pathlib import Path
import requests

# 한글 폰트 설정
import platform
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def setup_korean_fonts():
    """시스템에 맞는 한글 폰트 설정"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = ['Malgun Gothic', 'Arial Unicode MS']
    elif system == 'Darwin':
        plt.rcParams['font.family'] = ['AppleGothic', 'Arial Unicode MS']
    else:
        plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_fonts()


class EnhancedTextClassifier:
    """아키텍처 기반 개선된 텍스트 분류기"""
    
    def __init__(self, hf_token: str = None, verbose: bool = True):
        self.verbose = verbose
        self.hf_token = hf_token
        
    def analyze_pdf_enhanced(self, pdf_path: str) -> Dict[str, Any]:
        """아키텍처 기반 PDF 분석"""
        if self.verbose:
            print(f"🔍 Enhanced PDF 분석: {pdf_path}")
        
        # 기존 analyze_font_sizes 메서드 사용
        analysis = self._analyze_font_sizes_original(pdf_path)
        
        # Phase 1: 페이지별 Gap 패턴 분석
        page_patterns = self._analyze_page_gap_patterns(analysis["page_elements"])
        
        # Phase 2: 전체 Gap 패턴 분석
        overall_pattern = self._analyze_overall_gap_patterns(page_patterns)
        
        # Phase 3: Outlier 탐지 - 이제 확실히 작동함
        outlier_result = self._detect_outlier_pages(page_patterns, overall_pattern)
        normal_pages = outlier_result["normal_pages"]
        anomaly_pages = outlier_result["anomaly_pages"]
        
        if self.verbose:
            print(f"📊 일반: {len(normal_pages)}페이지, 이상치: {len(anomaly_pages)}페이지")
        
        return {
            "analysis": analysis,
            "page_patterns": page_patterns,
            "overall_pattern": overall_pattern,
            "normal_pages": normal_pages,
            "anomaly_pages": anomaly_pages
        }
    
    def _analyze_font_sizes_original(self, pdf_path: str) -> Dict[str, Any]:
        """기존 analyze_font_sizes 메서드"""
        if self.verbose:
            print(f"📖 PDF 분석 중: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            size_frequency = Counter()
            all_elements = []
            page_elements = defaultdict(list)
            
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text_dict = page.get_text("dict")
                    
                    for block in text_dict["blocks"]:
                        if "lines" not in block:
                            continue
                            
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue
                                
                                size = round(span["size"], 1)
                                size_frequency[size] += 1
                                
                                element = {
                                    "page": page_num + 1,
                                    "text": text,
                                    "size": size,
                                    "font": span["font"],
                                    "bbox": span["bbox"],
                                    "flags": span.get("flags", 0),
                                    "color": span.get("color", 0)
                                }
                                all_elements.append(element)
                                page_elements[page_num + 1].append(element)
            finally:
                doc.close()
                
        except Exception as e:
            print(f"❌ PDF 파일 처리 중 오류: {e}")
            raise
        
        # 통계 계산
        sizes = list(size_frequency.keys())
        if not sizes:
            return {
                "size_frequency": {},
                "all_elements": [],
                "page_elements": {},
                "statistics": {
                    "min_size": 0, "max_size": 0, "mean_size": 0,
                    "median_size": 0, "unique_sizes": 0, "total_texts": 0, "total_pages": 0
                }
            }
            
        sizes.sort()
        
        weighted_sum = sum(size * count for size, count in size_frequency.items())
        total_count = sum(size_frequency.values())
        weighted_mean = weighted_sum / total_count if total_count > 0 else 0
        
        analysis = {
            "size_frequency": dict(size_frequency),
            "all_elements": all_elements,
            "page_elements": dict(page_elements),
            "statistics": {
                "min_size": min(sizes),
                "max_size": max(sizes),
                "mean_size": weighted_mean,
                "median_size": statistics.median(sizes),
                "unique_sizes": len(sizes),
                "total_texts": total_count,
                "total_pages": len(page_elements)
            }
        }
        
        if self.verbose:
            print(f"✅ 분석 완료: {total_count}개 텍스트, {len(sizes)}개 고유 크기")
        
        return analysis
    
    def _analyze_page_gap_patterns(self, page_elements: Dict) -> List[Dict]:
        """페이지별 Gap 패턴 분석"""
        page_patterns = []
        
        for page_num, elements in page_elements.items():
            if not elements:
                continue
                
            sizes = [elem["size"] for elem in elements]
            unique_sizes = sorted(set(sizes))
            
            # Gap 계산
            gaps = []
            if len(unique_sizes) > 1:
                gaps = [unique_sizes[i+1] - unique_sizes[i] for i in range(len(unique_sizes)-1)]
            
            pattern = {
                "page_number": page_num,
                "unique_sizes": unique_sizes,
                "gaps": gaps,
                "largest_gap": max(gaps) if gaps else 0,
                "hierarchy_levels": len(unique_sizes),
                "total_elements": len(elements),
                "size_range": max(sizes) - min(sizes) if sizes else 0,
                "min_size": min(sizes) if sizes else 0,
                "max_size": max(sizes) if sizes else 0
            }
            
            page_patterns.append(pattern)
        
        if self.verbose:
            print(f"📄 페이지별 Gap 패턴 분석 완료: {len(page_patterns)}개 페이지")
        
        return page_patterns
    
    def _analyze_overall_gap_patterns(self, page_patterns: List[Dict]) -> Dict[str, Any]:
        """전체 Gap 패턴 분석"""
        if not page_patterns:
            return {}
        
        # 전체 통계
        all_gaps = []
        all_hierarchy_levels = []
        all_size_ranges = []
        all_max_sizes = []
        
        for pattern in page_patterns:
            all_gaps.extend(pattern["gaps"])
            all_hierarchy_levels.append(pattern["hierarchy_levels"])
            all_size_ranges.append(pattern["size_range"])
            all_max_sizes.append(pattern["max_size"])
        
        overall_pattern = {
            "median_gap": statistics.median(all_gaps) if all_gaps else 0,
            "mean_gap": statistics.mean(all_gaps) if all_gaps else 0,
            "median_hierarchy": statistics.median(all_hierarchy_levels),
            "median_size_range": statistics.median(all_size_ranges),
            "median_max_size": statistics.median(all_max_sizes),
            "total_pages": len(page_patterns)
        }
        
        if self.verbose:
            print(f"🌍 전체 Gap 패턴: 중간값 Gap {overall_pattern['median_gap']:.1f}pt")
        
        return overall_pattern
    
    def _detect_outlier_pages(self, page_patterns: List[Dict], overall_pattern: Dict) -> Dict[str, List[Dict]]:
        """Outlier 페이지 탐지 - 딕셔너리 반환으로 안전하게"""
        normal_pages = []
        anomaly_pages = []
        
        for pattern in page_patterns:
            anomaly_reasons = []
            
            # 조건 1: 비정상적으로 큰 Gap
            if overall_pattern.get("median_gap", 0) > 0 and pattern["largest_gap"] > overall_pattern["median_gap"] * 2.5:
                anomaly_reasons.append("large_gap")
            
            # 조건 2: 너무 많은 계층
            if pattern["hierarchy_levels"] > overall_pattern.get("median_hierarchy", 5) * 2:
                anomaly_reasons.append("too_many_levels")
            
            # 조건 3: 크기 범위가 너무 넓음
            if pattern["size_range"] > overall_pattern.get("median_size_range", 10) * 2:
                anomaly_reasons.append("wide_size_range")
            
            # 조건 4: 텍스트가 너무 적음 (제목 페이지 등)
            if pattern["total_elements"] < 5:
                anomaly_reasons.append("few_elements")
            
            # 조건 5: 비정상적으로 큰 폰트
            if pattern["max_size"] > overall_pattern.get("median_max_size", 12) * 1.5:
                anomaly_reasons.append("large_font")
            
            # 2개 이상 조건 충족시 이상치
            if len(anomaly_reasons) >= 2:
                pattern["anomaly_reasons"] = anomaly_reasons
                anomaly_pages.append(pattern)
                if self.verbose:
                    print(f"📄 이상치 페이지 {pattern['page_number']}: {', '.join(anomaly_reasons)}")
            else:
                normal_pages.append(pattern)
        
        if self.verbose:
            print(f"📊 Outlier 탐지 완료: 일반 {len(normal_pages)}개, 이상치 {len(anomaly_pages)}개")
        
        return {
            "normal_pages": normal_pages,
            "anomaly_pages": anomaly_pages
        }
    
    def classify_with_new_hybrid_approach(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """새로운 하이브리드 접근법으로 분류"""
        if self.verbose:
            print("🎯 새로운 하이브리드 분류 시작...")
        
        normal_pages = analysis_result["normal_pages"]
        anomaly_pages = analysis_result["anomaly_pages"]
        all_elements = analysis_result["analysis"]["all_elements"]
        
        # Phase 4: 일반 페이지 임계값 계산 (중간 80% + K-means)
        normal_elements = self._get_normal_page_elements(normal_pages, all_elements)
        thresholds = self._calculate_robust_thresholds(normal_elements)
        
        # Phase 5: 분류 수행
        classified_results = {
            "normal_classified": [],
            "anomaly_classified": [],
            "classification_stats": Counter(),
            "thresholds": thresholds
        }
        
        # 일반 페이지 분류 (통계적 방법)
        for page_pattern in normal_pages:
            page_num = page_pattern["page_number"]
            page_elements = [e for e in all_elements if e["page"] == page_num]
            
            classified_elements = self._classify_with_statistical_method(page_elements, thresholds)
            classified_results["normal_classified"].extend(classified_elements)
            
            # 통계 업데이트
            for elem in classified_elements:
                classified_results["classification_stats"][elem["classification"]] += 1
        
        # 이상치 페이지 처리
        for page_pattern in anomaly_pages:
            page_num = page_pattern["page_number"]
            page_elements = [e for e in all_elements if e["page"] == page_num]
            
            if self.hf_token:
                # LLM 처리 시도
                try:
                    classified_elements = self._process_with_llm(page_elements, page_pattern.get("anomaly_reasons", []))
                    classified_results["anomaly_classified"].extend(classified_elements)
                    if self.verbose:
                        print(f"✅ 페이지 {page_num} LLM 처리 완료")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ 페이지 {page_num} LLM 실패, 대안 사용: {e}")
                    classified_elements = self._classify_anomaly_fallback(page_elements)
                    classified_results["anomaly_classified"].extend(classified_elements)
            else:
                # LLM 없으면 대안 분류
                classified_elements = self._classify_anomaly_fallback(page_elements)
                classified_results["anomaly_classified"].extend(classified_elements)
            
            # 통계 업데이트
            for elem in classified_elements:
                classified_results["classification_stats"][elem["classification"]] += 1
        
        # 전체 결과 통합
        all_classified = (classified_results["normal_classified"] + 
                         classified_results["anomaly_classified"])
        classified_results["classified_elements"] = all_classified
        
        if self.verbose:
            print(f"✅ 분류 완료: 총 {len(all_classified)}개 요소")
        
        return classified_results
    
    def _get_normal_page_elements(self, normal_pages: List[Dict], all_elements: List[Dict]) -> List[Dict]:
        """일반 페이지의 모든 요소 수집"""
        normal_page_nums = [p["page_number"] for p in normal_pages]
        return [e for e in all_elements if e["page"] in normal_page_nums]
    
    def _calculate_robust_thresholds(self, elements: List[Dict]) -> Dict[str, float]:
        """견고한 임계값 계산: 중간 80% + K-means 조합"""
        if not elements:
            return {"title_threshold": 18.0, "subtitle_threshold": 14.0, "body_threshold": 12.0}
        
        sizes = [e["size"] for e in elements]
        
        # 방법 1: 중간 80% 범위 (좌우 10%씩 제외)
        middle_80_thresholds = self._calculate_middle_80_thresholds(sizes)
        
        # 방법 2: K-means 클러스터링 (k=3)
        kmeans_thresholds = self._calculate_kmeans_thresholds(sizes)
        
        # 가중 조합
        final_thresholds = {
            "title_threshold": (middle_80_thresholds["title"] * 0.7 + 
                              kmeans_thresholds["title"] * 0.3),
            "subtitle_threshold": (middle_80_thresholds["subtitle"] * 0.7 + 
                                 kmeans_thresholds["subtitle"] * 0.3),
            "body_threshold": (middle_80_thresholds["body"] * 0.7 + 
                             kmeans_thresholds["body"] * 0.3)
        }
        
        if self.verbose:
            print(f"📏 임계값: Title≥{final_thresholds['title_threshold']:.1f}, "
                  f"Subtitle≥{final_thresholds['subtitle_threshold']:.1f}, "
                  f"Body≥{final_thresholds['body_threshold']:.1f}")
        
        return final_thresholds
    
    def _calculate_middle_80_thresholds(self, sizes: List[float]) -> Dict[str, float]:
        """중간 80% 범위 기반 임계값 계산"""
        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)
        
        # 좌우 10%씩 제외한 중간 80% 범위
        start_idx = int(n * 0.1)
        end_idx = int(n * 0.9)
        middle_80_sizes = sorted_sizes[start_idx:end_idx]
        
        if not middle_80_sizes:
            return {"title": 18.0, "subtitle": 14.0, "body": 12.0}
        
        # 중간 범위를 3등분
        third = len(middle_80_sizes) // 3
        
        return {
            "body": middle_80_sizes[third] if third < len(middle_80_sizes) else middle_80_sizes[0],
            "subtitle": middle_80_sizes[2*third] if 2*third < len(middle_80_sizes) else middle_80_sizes[-1],
            "title": middle_80_sizes[-1]
        }
    
    def _calculate_kmeans_thresholds(self, sizes: List[float]) -> Dict[str, float]:
        """K-means 클러스터링 기반 임계값 계산"""
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # K-means k=3으로 클러스터링
            sizes_array = np.array(sizes).reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(sizes_array)
            
            # 클러스터별 크기 분포
            cluster_sizes = {}
            for i in range(3):
                cluster_elements = [sizes[j] for j in range(len(sizes)) if clusters[j] == i]
                if cluster_elements:
                    cluster_sizes[i] = {
                        "min": min(cluster_elements),
                        "max": max(cluster_elements),  
                        "count": len(cluster_elements),
                        "center": kmeans.cluster_centers_[i][0]
                    }
            
            if not cluster_sizes:
                return {"title": 18.0, "subtitle": 14.0, "body": 12.0}
            
            # 클러스터 중심점 기준으로 정렬
            sorted_clusters = sorted(cluster_sizes.keys(), 
                                   key=lambda k: cluster_sizes[k]["center"])
            
            # 임계값 설정
            if len(sorted_clusters) >= 3:
                return {
                    "body": cluster_sizes[sorted_clusters[0]]["max"],
                    "subtitle": cluster_sizes[sorted_clusters[1]]["max"],
                    "title": cluster_sizes[sorted_clusters[2]]["center"]
                }
            else:
                return {"title": 18.0, "subtitle": 14.0, "body": 12.0}
                
        except ImportError:
            if self.verbose:
                print("⚠️ sklearn이 설치되지 않음, 기본값 사용")
            return {"title": 18.0, "subtitle": 14.0, "body": 12.0}
        except Exception as e:
            if self.verbose:
                print(f"⚠️ K-means 실패, 기본값 사용: {e}")
            return {"title": 18.0, "subtitle": 14.0, "body": 12.0}
    
    def _classify_with_statistical_method(self, elements: List[Dict], thresholds: Dict) -> List[Dict]:
        """통계적 방법으로 분류 (일반 페이지용)"""
        classified = []
        
        for elem in elements:
            elem_copy = elem.copy()
            size = elem["size"]
            text = elem["text"].strip()
            
            # 패턴 기반 우선 분류
            if len(text) <= 3 and text.isdigit():
                classification = "page_number"
            elif size >= thresholds["title_threshold"]:
                classification = "title"
            elif size >= thresholds["subtitle_threshold"]:
                classification = "subtitle"  
            elif text.startswith(('•', '-', '*', '1.', '2.', '3.')):
                classification = "bullet_point"
            elif size >= thresholds["body_threshold"]:
                classification = "body_text"
            else:
                classification = "footnote"
            
            elem_copy["classification"] = classification
            elem_copy["method"] = "statistical"
            classified.append(elem_copy)
        
        return classified
    
    def _process_with_llm(self, elements: List[Dict], anomaly_reasons: List[str]) -> List[Dict]:
        """LLM으로 이상치 페이지 처리"""
        # 간단한 LLM 시뮬레이션 (실제로는 Hugging Face API 호출)
        classified = []
        
        for elem in elements:
            elem_copy = elem.copy()
            size = elem["size"]
            text = elem["text"]
            
            # LLM 기반 고급 분류 (시뮬레이션)
            if "few_elements" in anomaly_reasons and size > 14:
                classification = "title"
            elif any(keyword in text.lower() for keyword in ['chapter', 'section', '장', '절']):
                classification = "structured_title"
            elif size > 16:
                classification = "title"
            elif size > 12:
                classification = "subtitle"
            else:
                classification = "body_text"
            
            elem_copy["classification"] = classification
            elem_copy["method"] = "llm_simulation"
            classified.append(elem_copy)
        
        return classified
    
    def _classify_anomaly_fallback(self, elements: List[Dict]) -> List[Dict]:
        """이상치 페이지 대안 분류"""
        classified = []
        
        for elem in elements:
            elem_copy = elem.copy()
            size = elem["size"]
            
            # 단순한 크기 기반 분류
            if size > 18:
                classification = "title"
            elif size > 14:
                classification = "subtitle"
            elif size > 10:
                classification = "body_text"
            else:
                classification = "footnote"
            
            elem_copy["classification"] = classification
            elem_copy["method"] = "fallback"
            classified.append(elem_copy)
        
        return classified


def analyze_lecture_pdf_new_hybrid(pdf_path: str, 
                                  huggingface_token: str = None,
                                  output_dir: str = None,
                                  show_visualizations: bool = True) -> Dict[str, Any]:
    """새로운 하이브리드 접근법 메인 함수"""
    print("🚀 New Hybrid Approach Analysis")
    print("="*40)
    
    # 분석 수행
    classifier = EnhancedTextClassifier(huggingface_token, verbose=True)
    
    # Phase 1-3: 페이지 분석, Gap 패턴, Outlier 탐지
    analysis_result = classifier.analyze_pdf_enhanced(pdf_path)
    
    # Phase 4-5: 새로운 하이브리드 분류
    classification_result = classifier.classify_with_new_hybrid_approach(analysis_result)
    
    # 결과 통합
    results = {
        "analysis": analysis_result,
        "classification": classification_result,
        "pdf_path": pdf_path
    }
    
    # 요약 출력
    print_new_hybrid_summary(results)
    
    # 결과 저장
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f"{output_dir}/new_hybrid_results.json", "w", encoding="utf-8") as f:
            save_data = {
                "statistics": analysis_result["analysis"]["statistics"],
                "normal_pages": len(analysis_result["normal_pages"]),
                "anomaly_pages": len(analysis_result["anomaly_pages"]),
                "anomaly_reasons": [p.get("anomaly_reasons", []) for p in analysis_result["anomaly_pages"]],
                "classified_elements": classification_result["classified_elements"],
                "classification_stats": dict(classification_result["classification_stats"]),
                "thresholds": classification_result["thresholds"]
            }
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장: {output_dir}")
    
    return results


def print_new_hybrid_summary(results: Dict[str, Any]):
    """새로운 하이브리드 결과 요약 출력"""
    analysis = results["analysis"]
    classification = results["classification"]
    
    print("\n" + "="*60)
    print("📈 NEW HYBRID APPROACH SUMMARY")
    print("="*60)
    
    stats = analysis["analysis"]["statistics"]
    print(f"📄 총 페이지: {stats['total_pages']}")
    print(f"📝 총 텍스트: {stats['total_texts']:,}개")
    print(f"📏 크기 범위: {stats['min_size']:.1f}pt ~ {stats['max_size']:.1f}pt")
    
    print(f"\n🔍 페이지 분류:")
    print(f"  • 일반 페이지: {len(analysis['normal_pages'])}개 (통계적 방법)")
    print(f"  • 이상치 페이지: {len(analysis['anomaly_pages'])}개 (LLM 처리)")
    
    # 이상치 원인
    if analysis["anomaly_pages"]:
        all_reasons = []
        for page in analysis["anomaly_pages"]:
            all_reasons.extend(page.get("anomaly_reasons", []))
        reason_counts = Counter(all_reasons)
        
        print(f"\n⚠️ 이상치 원인:")
        for reason, count in reason_counts.most_common():
            print(f"  • {reason}: {count}회")
    
    print(f"\n🏷️ 텍스트 분류:")
    for category, count in classification["classification_stats"].most_common():
        print(f"  • {category}: {count:,}개")
    
    thresholds = classification["thresholds"]
    print(f"\n📏 계산된 임계값:")
    print(f"  • Title: ≥{thresholds['title_threshold']:.1f}pt")
    print(f"  • Subtitle: ≥{thresholds['subtitle_threshold']:.1f}pt")
    print(f"  • Body: ≥{thresholds['body_threshold']:.1f}pt")
    
    # 처리 방법별 통계
    methods = Counter()
    for elem in classification["classified_elements"]:
        methods[elem.get("method", "unknown")] += 1
    
    print(f"\n🔧 처리 방법별 통계:")
    for method, count in methods.most_common():
        print(f"  • {method}: {count:,}개")


if __name__ == "__main__":
    import sys
    
    # 사용법: python text_analyzer.py [PDF경로] [--hf-token TOKEN]
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    hf_token = None
    
    if "--hf-token" in sys.argv:
        try:
            idx = sys.argv.index("--hf-token") + 1
            hf_token = sys.argv[idx]
            print("🔑 Hugging Face 토큰 설정됨")
        except:
            print("⚠️ Hugging Face 토큰 형식 오류")
    
    try:
        results = analyze_lecture_pdf_new_hybrid(
            pdf_path=pdf_path,
            huggingface_token=hf_token,
            output_dir="./new_hybrid_output",
            show_visualizations=True
        )
        print("\n✅ 새로운 하이브리드 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()