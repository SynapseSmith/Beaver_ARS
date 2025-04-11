import random
import pandas as pd

# 1. 사용자 발화 예시 리스트 생성
user_utterances = [
    "오늘 짜장면 가격 얼마야?",
    "카카오페이로 결제 가능해?",
    "화요일에 영업해?",
    "돈까스는 어떤 소스로 제공돼?",
    "현금 외에 다른 결제수단 있어?",
    "금요일에 할인 행사 있나요?",
    "스파게티와 피자 메뉴가 궁금해.",
    "신용카드 결제는 되나요?",
    "토요일에 오픈 시간이 어떻게 돼?",
    "라면 가격 좀 알려줘.",
    "네이버 페이로도 결제할 수 있나요?",
    "일요일에 특별 메뉴가 있나요?",
    "후라이드 치킨은 어떤 소스와 함께 제공돼?",
    "모바일 결제 지원하나요?",
    "수요일에 예약 가능한가요?",
    "불고기 버거의 가격은?",
    "페이코로 결제하려면 어떻게 해야 하나요?",
    "목요일에만 판매되는 메뉴가 있나요?",
    "현금 결제 시 할인 혜택이 있나요?",
    "저녁 7시에 영업 종료돼?"ahs
]

# 2. DataFrame으로 변환
df_utterances = pd.DataFrame(user_utterances, columns=['utterance'])

# 3. 데이터 확인
print("생성된 사용자 발화 데이터:")
print(df_utterances.head(10))




# 추가적인 엔티티 예시 리스트
menus = ["짜장면", "김치찌개", "돈까스", "스파게티", "피자", "라면", "후라이드 치킨", "불고기 버거"]
payments = ["카카오페이", "현금", "신용카드", "네이버 페이", "페이코", "모바일 결제"]
days = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

# 템플릿 리스트
templates = [
    "오늘 {menu} 가격 얼마야?",
    "{payment}으로 결제 가능해?",
    "{day}에 영업해?",
    "{menu}는 어떤 소스로 제공돼?",
    "현금 외에 다른 결제수단 있어?",
    "{day}에 할인 행사 있나요?",
    "{menu}와 피자 메뉴가 궁금해.",
    "신용카드 결제는 되나요?",
    "{day}에 오픈 시간이 어떻게 돼?",
    "{menu} 가격 좀 알려줘.",
    "네이버 페이로도 결제할 수 있나요?",
    "{day}에 특별 메뉴가 있나요?",
    "{menu}는 어떤 소스와 함께 제공돼?",
    "모바일 결제 지원하나요?",
    "{day}에 예약 가능한가요?",
    "{menu}의 가격은?",
    "{payment}로 결제하려면 어떻게 해야 하나요?",
    "{day}에만 판매되는 메뉴가 있나요?",
    "현금 결제 시 할인 혜택이 있나요?",
    "저녁 7시에 영업 종료돼?"
]

# 4. 데이터 확장 함수
def generate_utterances(num_samples=100):
    generated_utterances = []
    for _ in range(num_samples):
        template = random.choice(templates)
        menu = random.choice(menus)
        payment = random.choice(payments)
        day = random.choice(days)
        utterance = template.format(menu=menu, payment=payment, day=day)
        generated_utterances.append(utterance)
    return generated_utterances

# 5. 추가 데이터 생성
additional_utterances = generate_utterances(80)  # 80개의 추가 발화 생성

# 6. DataFrame에 추가
df_additional = pd.DataFrame(additional_utterances, columns=['utterance'])
df_all = pd.concat([df_utterances, df_additional], ignore_index=True)

# 7. 최종 데이터 확인
print("최종 사용자 발화 데이터:")
print(df_all.head(10))
print(f"총 발화 수: {len(df_all)}")


# 4. 데이터 저장
df_all.to_csv("/home/user09/beaver/data/shared_files/241218_NER/data/step1_example_ner_data.csv")