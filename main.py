# -*- coding: utf-8 -*-
"""
하나의 파일에서 1차(직접 욕설) + 2차(간접 비방) 분류를 수행하고, 단일 CSV로 결과를 저장합니다.
- 입력 CSV는 최소한 ['title', 'link', 'content_excerpt'] 컬럼을 가진다고 가정합니다.
- 텍스트는 'content'가 있으면 우선 사용하고, 없으면 'content_excerpt'를 사용합니다.
- 1차가 '욕설 사용'이면 2차는 비용 절약을 위해 건너뜁니다(컬럼에 '미실행' 표기).

사용 예시:
  GEMINI_API=YOUR_KEY \
  python combined_direct_indirect_classifier.py \
    --input ./data/crawled_posts_basic.csv \
    --output ./data/classified_direct_indirect.csv \
    --model-direct gemini-2.0-flash \
    --model-indirect gemini-2.0-flash \
    --sleep-min 1.0 --sleep-max 2.0

필요 패키지:
  pip install python-dotenv langchain langchain-google-genai tqdm beautifulsoup4 requests
"""
from __future__ import annotations
import os
import csv
import time
import random
import argparse
from typing import Dict, Any, List

from dotenv import load_dotenv
from tqdm import tqdm

# LangChain / Google GenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field

# -----------------------------
# 0) 공통: 설정/LLM 초기화
# -----------------------------

def build_llm(api_key: str, model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model_name,
        temperature=temperature,
        top_p=1,
        top_k=1,
    )

# -----------------------------
# 1) 데이터 로드
# -----------------------------

def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# -----------------------------
# 2) 1차: 직접 욕설 사용 여부(네 코드 프롬프트 기반)
# -----------------------------
class DirectProfanityResult(BaseModel):
    classification: str = Field(description="직접 욕설이 있으면 '욕설 사용', 없으면 '일반글'")
    reason: str = Field(description="판단 이유")
    confidence_score: float = Field(description="0.0~1.0")

DIRECT_PROMPT_TMPL = """
당신은 한국어 텍스트에서 명백하고 직접적인 욕설의 사용 여부를 판단하는 정확한 언어 분석 전문가입니다.
다른 종류의 비방, 비판, 풍자, 은어, 간접적인 표현이 아닌, **오직 직접적으로 사용된 욕설 단어 자체의 존재 유무**에만 집중해주세요.

[판단 지침]
1. **직접성**: 텍스트에 실제 욕설로 분류될 수 있는 단어가 직접적으로 쓰였는지 확인합니다(예: "씨발", "개새끼", "병신" 등).
2. **명확성**: 누구나 욕설로 인지할 수 있는 명확한 단어에 초점을 맞춥니다. 그 외는 '욕설 사용'으로 보지 않습니다.
3. **제외**: 비유/풍자/암시/은어는 고려하지 않습니다. 여기서는 오직 직접 욕설 단어만 봅니다.

[게시글]
제목: "{post_title}"
내용:
---
{text_content}
---

출력(JSON 스키마):
{format_instructions}

[예시 출력 가이드]
- 욕설이면: classification="욕설 사용", reason에 감지 단어를 간단히, confidence_score≈0.9~1.0
- 아니면: classification="일반글", reason="직접적인 욕설 없음", confidence_score는 그에 맞게.
"""

def run_direct(llm_direct: ChatGoogleGenerativeAI, title: str, text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"classification": "분석 불가", "reason": "내용 없음", "confidence": 0.0}
    parser = JsonOutputParser(pydantic_object=DirectProfanityResult)
    prompt = ChatPromptTemplate.from_template(
        template=DIRECT_PROMPT_TMPL,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm_direct
    try:
        raw = chain.invoke({"post_title": title, "text_content": text})
        parsed = parser.parse(raw.content)
        return {
            "classification": parsed.get("classification", "파싱 오류"),
            "reason": parsed.get("reason", ""),
            "confidence": float(parsed.get("confidence_score", 0.0) or 0.0),
        }
    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            return {"classification": "분석 실패 (API 할당량)", "reason": msg[:120], "confidence": 0.0}
        return {"classification": "분석 실패", "reason": msg[:120], "confidence": 0.0}

# -----------------------------
# 3) 2차: 간접 비방(비유/풍자/암시)
# -----------------------------
class IndirectDefamationResult(BaseModel):
    is_defamatory: bool = Field(description="간접 비방이면 true, 아니면 false")
    reason: str = Field(description="판단 이유")
    confidence_score: float = Field(description="0.0~1.0")

INDIRECT_PROMPT_TMPL = """
당신은 커뮤니티 게시글을 분석하여, **직접 욕설 여부는 무시**하고, **간접적 방법(비유, 풍자, 은어, 암시 등)으로 특정 대상을 공격/조롱/명예훼손하려는 의도**가 있는지 판단한다.
단순 감정 표현이나 중립적 비교는 간접 비방으로 보지 않는다.

[판단 기준] (하나 이상 충족 시 간접 비방)
1) 특정 대상에 대한 악의적 조롱/희화화가 비유/암시로 표현됨
2) 최근 사회적 이슈/밈/사건/인물에 빗대어 상대를 낮추는 표현
3) 은어/우회표현을 통해 모욕의도를 전달(대상·의도 추론 가능)

[게시글]
제목: "{post_title}"
내용:
---
{text_content}
---

출력(JSON 스키마):
{format_instructions}

[예시 출력 가이드]
- 간접 비방이면 is_defamatory=true, reason에 핵심 근거 구절/표현을 요약, confidence_score≈0.7~1.0
- 아니면 is_defamatory=false, reason에 간단 사유.
"""

def run_indirect(llm_indirect: ChatGoogleGenerativeAI, title: str, text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"classification": "분석 불가", "reason": "내용 없음", "confidence": 0.0}
    parser = JsonOutputParser(pydantic_object=IndirectDefamationResult)
    prompt = ChatPromptTemplate.from_template(
        template=INDIRECT_PROMPT_TMPL,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm_indirect | parser
    try:
        res = chain.invoke({"post_title": title, "text_content": text})
        return {
            "classification": "간접 비방" if bool(res.get("is_defamatory")) else "일반글",
            "reason": res.get("reason", ""),
            "confidence": float(res.get("confidence_score", 0.0) or 0.0),
        }
    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            return {"classification": "분석 실패 (API 할당량)", "reason": msg[:120], "confidence": 0.0}
        return {"classification": "분석 실패", "reason": msg[:120], "confidence": 0.0}

# -----------------------------
# 4) 메인: 통합 실행
# -----------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        print("오류: GEMINI_API 환경 변수를 설정하세요.")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='입력 CSV 경로')
    ap.add_argument('--output', required=True, help='출력 CSV 경로')
    ap.add_argument('--model-direct', default='gemini-2.0-flash', help='1차 모델명')
    ap.add_argument('--model-indirect', default='gemini-2.0-flash', help='2차 모델명')
    ap.add_argument('--sleep-min', type=float, default=1.0, help='호출 사이 최소 대기초')
    ap.add_argument('--sleep-max', type=float, default=2.0, help='호출 사이 최대 대기초')
    args = ap.parse_args()

    llm_direct = build_llm(api_key, args.model_direct, temperature=0.3)
    llm_indirect = build_llm(api_key, args.model_indirect, temperature=0.7)

    rows = load_rows(args.input)
    if not rows:
        print("입력 데이터가 없습니다.")
        return

    fieldnames = [
        "title", "link", "content_excerpt",
        "direct_classification", "direct_reason", "direct_confidence",
        "indirect_classification", "indirect_reason", "indirect_confidence",
        "final_label", "final_reason"
    ]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="분류 진행"):
            title = row.get('title', '')
            link = row.get('link', '')
            # content > content_excerpt 우선
            text = row.get('content') or row.get('content_excerpt') or ''

            # 1차(직접 욕설)
            dres = run_direct(llm_direct, title, text)
            # 2차(간접 비방): 1차가 욕설이면 스킵
            if dres.get('classification') == '욕설 사용':
                ires = {"classification": "미실행", "reason": "1차에서 욕설 확정", "confidence": 0.0}
            else:
                ires = run_indirect(llm_indirect, title, text)

            # 최종 라벨/사유
            if dres.get('classification') == '욕설 사용':
                final_label = '욕설 사용'
                final_reason = f"1차 확정: {dres.get('reason','')}"
            elif ires.get('classification') == '간접 비방':
                final_label = '간접 비방'
                final_reason = f"2차 확정: {ires.get('reason','')}"
            else:
                final_label = '일반글'
                final_reason = '직접 욕설 및 간접 비방 모두 아님'

            writer.writerow({
                "title": title,
                "link": link,
                "content_excerpt": text[:200].replace('\n', ' ') + '...' if isinstance(text, str) else '내용 없음',
                "direct_classification": dres.get('classification',''),
                "direct_reason": dres.get('reason',''),
                "direct_confidence": f"{float(dres.get('confidence',0.0)):.2f}",
                "indirect_classification": ires.get('classification',''),
                "indirect_reason": ires.get('reason',''),
                "indirect_confidence": f"{float(ires.get('confidence',0.0)):.2f}",
                "final_label": final_label,
                "final_reason": final_reason,
            })

            # API 레이트 한도 배려
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    print(f"완료: {args.output}")


if __name__ == '__main__':
    main()
