import pandas as pd
import re

# 1. 라벨링 함수 정의
def label_entities(utterance):
    # 초기화: 모든 토큰을 'O'로 설정
    tokens = utterance.split()
    labels = ['O'] * len(tokens)
    
    # 엔티티 리스트
    entities = {
        'MENU': ["짜장면", "김치찌개", "돈까스", "스파게티", "피자", "라면", "후라이드 치킨", "불고기 버거"],
        'PAYMENT': ["카카오페이", "현금", "신용카드", "네이버 페이", "페이코", "모바일 결제"],
        'DAY': ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    }
    
    # 각 엔티티에 대해 매칭
    for entity, keywords in entities.items():
        for keyword in keywords:
            # 다중 토큰 엔티티 지원
            pattern = re.compile(re.escape(keyword))
            for match in pattern.finditer(utterance):
                start, end = match.span()
                # 토큰 위치 찾기
                current_pos = 0
                entity_tokens = keyword.split()
                for i, token in enumerate(tokens):
                    token_start = utterance.find(token, current_pos)
                    token_end = token_start + len(token)
                    current_pos = token_end
                    if token_start == -1:
                        continue  # 토큰을 찾을 수 없는 경우 건너뜀
                    # 엔티티의 시작 토큰 위치인지 확인
                    if token_start <= start < token_end:
                        labels[i] = f'B-{entity}'
                        # 다중 토큰 엔티티 처리
                        for j in range(1, len(entity_tokens)):
                            if i + j < len(tokens):
                                labels[i + j] = f'I-{entity}'
                        break
    return list(zip(tokens, labels))

# 2. 라벨링 적용
# 사용자 발화 데이터를 직접 생성 (CSV 파일 경로가 제공되지 않아 예시로 대체)
data = {
    "utterance": [
        "이 식당에서 불고기 버거를 판매하나요?",
        "오늘 스파게티 메뉴가 있나요?",
        "메뉴에 치즈케이크가 포함되어 있나요?",
        "라멘을 주문할 수 있나요?",
        "이 카페에서 아메리카노를 제공하나요?",
        "오늘 특별 메뉴로 랍스터가 있나요?",
        "피자에 해산물이 들어가 있나요?",
        "메뉴에 비건 옵션이 있나요?",
        "오늘 점심 세트 메뉴는 무엇이 있나요?",
        "디저트로 아이스크림을 판매하나요?",
        "이 곳에서 글루텐 프리 빵을 구매할 수 있나요?",
        "메뉴에 새우 요리가 포함되어 있나요?",
        "오늘은 햄버거 세트가 있나요?",
        "이 레스토랑에서 샐러드를 제공하나요?",
        "메뉴에 티라미수가 있나요?",
        "오늘 저녁에 스테이크를 주문할 수 있나요?",
        "이 카페에서 라떼를 판매하나요?",
        "메뉴에 초밥이 포함되어 있나요?",
        "오늘은 어떤 파스타가 있나요?",
        "이 식당에서 샌드위치를 제공하나요?",
        "메뉴에 글루텐 프리 피자가 있나요?",
        "오늘 특별히 제공되는 디저트가 있나요?",
        "이 곳에서 소고기 요리를 주문할 수 있나요?",
        "메뉴에 유제품 없는 옵션이 있나요?",
        "오늘은 해산물 파스타를 판매하나요?",
        "이 레스토랑에서 글루텐 프리 빵을 제공하나요?",
        "메뉴에 닭고기 요리가 포함되어 있나요?",
        "오늘 저녁에 제공되는 스시 메뉴는 무엇인가요?",
        "이 카페에서 디카페인 커피를 판매하나요?",
        "메뉴에 채식주의자용 샐러드가 있나요?",
        "오늘은 어떤 스프를 제공하나요?"
    ]
}

df = pd.DataFrame(data)
df['labels'] = df['utterance'].apply(label_entities)

# 3. 라벨링된 데이터 확인
print("라벨링된 데이터 예시:")
print(df[['utterance', 'labels']].head(5))

# 4. CoNLL 형식으로 데이터 저장 함수
def convert_to_conll(df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            labels = row['labels']
            for token, label in labels:
                f.write(f"{token}\t{label}\n")
            f.write("\n")  # 문장 구분을 위해 빈 줄 추가

# 5. CoNLL 파일로 저장
# 파일 경로을 환경에 맞게 수정하세요
convert_to_conll(df, '/home/user09/beaver/data/shared_files/241218_NER/data/step1_labeled_data.conll')

print("CoNLL 형식의 라벨링된 데이터가 'step1_labeled_data.conll'에 저장되었습니다.")

# 6. 일부 데이터 출력
for i in range(3):
    print(f"Utterance: {df.iloc[i]['utterance']}")
    for token, label in df.iloc[i]['labels']:
        print(f"{token}\t{label}")
    print("\n")
