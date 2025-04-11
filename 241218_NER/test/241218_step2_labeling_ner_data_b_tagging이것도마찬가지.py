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
            pattern = re.compile(re.escape(keyword))
            for match in pattern.finditer(utterance):
                start, end = match.span()
                # 토큰 위치 찾기
                current_pos = 0
                for i, token in enumerate(tokens):
                    token_start = utterance.find(token, current_pos)
                    token_end = token_start + len(token)
                    current_pos = token_end
                    if token_start <= start < token_end:
                        # B-ENTITY
                        labels[i] = f'B-{entity}'
                        # Check if the entity spans multiple tokens
                        # (Not applicable in current examples, but included for extensibility)
                        break
    return list(zip(tokens, labels))

# 2. 라벨링 적용
df = pd.read_csv("/home/user09/beaver/data/shared_files/241218_NER/data/step1_example_ner_data.csv")
df['labels'] = df['utterance'].apply(label_entities)

# 3. 라벨링된 데이터 확인
print("라벨링된 데이터 예시:")
print(df[['utterance', 'labels']].head(5))


# 1. CoNLL 형식으로 데이터 저장 함수
def convert_to_conll(df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            labels = row['labels']
            for token, label in labels:
                f.write(f"{token}\t{label}\n")
            f.write("\n")  # 문장 구분을 위해 빈 줄 추가

# 2. CoNLL 파일로 저장
convert_to_conll(df, '/home/user09/beaver/data/shared_files/241218_NER/data/step1_labeled_data.conll')

print("CoNLL 형식의 라벨링된 데이터가 'step1_labeled_data.conll'에 저장되었습니다.")



# 1. 일부 데이터 출력
for i in range(3):
    print(f"Utterance: {df.iloc[i]['utterance']}")
    for token, label in df.iloc[i]['labels']:
        print(f"{token}\t{label}")
    print("\n")