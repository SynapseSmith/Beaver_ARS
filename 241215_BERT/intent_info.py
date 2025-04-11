import pandas as pd

# 의도 별 숫자와 이름을 매핑한 딕셔너리
intents_dict = {
    0: "메뉴 카테고리 안내",
    1: "인기 / 추천",
    2: "계절 한정 메뉴 (예: 여름 특선, 겨울 메뉴) / 프로모션 메뉴 (예: 신메뉴 할인) 안내 / 신메뉴",
    3: "주문 / 전달 방식 안내 (키오스크, 테이블오더, 스마트주문 / 내점, 포장, 배달 등)",
    4: "결제 방법 안내(현금, 카드, 간편 결제등)",
    5: "영업 시작/종료 시간 안내",
    6: "정기 휴무/주말/공휴일 운영 여부 안내",
    7: "배달 가능 지역 안내",
    8: "배달비 및 최소 주문 금액 안내",
    9: "예약 가능 여부 안내(전화/온라인)",
    10: "예약 취소/변경 절차 안내",
    11: "고객 대기 및 혼잡도 안내",
    12: "상점 주소 안내 및 지도 링크 전달",
    13: "테이블 배치 및 좌석 수 안내",
    14: "야외 테라스 또는 개별 룸 여부 안내",
    15: "멤버십 가입 / 혜택 안내",
    16: "포인트 적립 / 사용 관련 안내",
    17: "쿠폰 발행 / 사용 안내",
    18: "현재 진행 중인 이벤트 및 할인 혜택, 기간 정보 안내",
    19: "상점 관리자 연결 안내",
    20: "fallbackintent"
}

# 의도 별 응대 템플릿
response_templates = {
    0: [
        "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 메뉴를 제공합니다. 자세한 사항은 문자로 전송드린 메뉴판을 확인해주세요.",
        "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 다양한 메뉴가 준비되어 있습니다.",
        "저희 {STORE_NM}에는 {STD_CATEGORY_NM}와 같은 메뉴들이 있습니다. 자세한 내용은 문자로 전송드린 메뉴판을 참조해주세요.",
        "{STORE_NM}에서는 {STD_CATEGORY_NM}와 같은 메뉴를 판매하고 있습니다."
    ],
    1: [
        "저희 {STORE_NM}의 {INDI_TYPE_NM1} 메뉴는 {RECOMMEND_MENU} 메뉴로서 추천드립니다.",
            "{STORE_NM}에서 {INDI_TYPE_NM1} 메뉴로서 {RECOMMEND_MENU} 메뉴를 추천드립니다.",
        "{STORE_NM}에서 고객님들이 가장 많이 찾는 메뉴는 {RECOMMEND_MENU} 입니다.",
    ],
    2: [
        "저희 매장에서 {INDI_TYPE_NM3} 메뉴는 {LIMMIT_MENU}입니다.",
        "저희 {STORE_NM}에서는 {LIMMIT_MENU} 메뉴를 {INDI_TYPE_NM3} 이벤트중입니다.",
        "지금 {STORE_NM}에서는 {INDI_TYPE_NM3} 이벤트 중인 {LIMMIT_MENU} 메뉴를 만나보실 수 있습니다.",
        "저희 매장에서는 {LIMMIT_MENU} 메뉴를 {INDI_TYPE_NM3} 이벤트로 선보이고 있습니다."
    ],
    3: [
        "저희 {STORE_NM}은 {ORDER_TYPE} 방식으로 주문이 가능해요.",
        "{STORE_NM}에서는 {ORDER_TYPE} 의 방법으로 주문 하실 수 있습니다.",
        "저희 {STORE_NM}에서 {ORDER_TYPE} 방식으로 주문이 가능합니다.",
        "{ORDER_TYPE} 방식을 통해 {STORE_NM}의 서비스를 이용하실 수 있습니다."
    ],
    4: [
        "저희 {STORE_NM}에서는 {PAYMNT_MN_CD} 방식의 결제가 가능하며 간편결제로는 {EASY_PAYMNT_TYPE_CD} 가능합니다.",
        "{STORE_NM}에서는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD} 방식을 지원합니다.",
        "결제는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD} 방식이 가능하니 참고해주세요.",
        "{STORE_NM}에서는 {PAYMNT_MN_CD} 방식의 결제가 가능합니다.",
    ],
    5: [
        "저희 {STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지에요.",
        "저희 {STORE_NM}은 {SALE_BGN_TM}부터 {SALE_END_TM}까지 운영됩니다.",
        "{STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지입니다.",
        "운영 시간은 {STORE_NM}에서 {SALE_BGN_TM}부터 {SALE_END_TM}까지입니다.",
        "{STORE_NM}은 {SALE_BGN_TM}부터 {SALE_END_TM}까지 영업합니다."
    ],
    6: [
        "저희 {STORE_NM}의 휴일은 {HOLIDAY_TYPE_CD}이에요.",
        "{STORE_NM}은 {HOLIDAY_TYPE_CD}에 휴무일입니다.",
        "{HOLIDAY_TYPE_CD}에 {STORE_NM}은 운영하지 않습니다.",
        "{STORE_NM}의 정기 휴무일은 {HOLIDAY_TYPE_CD}입니다."
    ],
    7: [
        "저희 {STORE_NM}은 반경 {DLVR_ZONE_RADS} 이내의 지역만 배달 가능합니다. 자세한 사항은 매장에 문의해주세요.",
        "{STORE_NM}은 반경 {DLVR_ZONE_RADS} 이내의 지역으로만 배달 서비스를 제공하고 있습니다. 자세한 내용은 매장에 문의해주세요.",
        "현재 {STORE_NM}의 배달 가능 지역은 반경 {DLVR_ZONE_RADS} 이내입니다. 자세한 내용은 매장으로 문의해주세요.",
        "저희 {STORE_NM}의 배달 범위는 반경 {DLVR_ZONE_RADS} 이내 지역입니다. 자세한 사항은 확인 후 안내 드리겠습니다.",
        "{STORE_NM}의 배달 서비스는 매장 반경 {DLVR_ZONE_RADS} 이내에서만 운영됩니다. 해당 부분은 확인 후 다시 안내드리겠습니다."
    ]
    ,
    8: [
        "저희 {STORE_NM}의 배달 팁은 최소 {DLVR_TIP_AMT} 이며 {ORDER_BGN_AMT} 이상 주문해주셔야 배달이 가능합니다.",
            "저희 {STORE_NM}은 배달 팁이 최소 {DLVR_TIP_AMT}이며, {ORDER_BGN_AMT} 이상 주문 시 배달 가능합니다.",
        "배달 팁은 최소 {DLVR_TIP_AMT}이고, {STORE_NM}에서는 {ORDER_BGN_AMT} 이상 주문해 주셔야 배달이 가능합니다.",
        "저희 {STORE_NM}에서 배달을 원하실 경우 최소 {ORDER_BGN_AMT} 이상 주문 시 배달이 가능하며, 배달 팁은 최소{DLVR_TIP_AMT}입니다.",
        "현재 {STORE_NM}의 배달 팁은 최소 {DLVR_TIP_AMT}이며, 최소 주문 금액은 {ORDER_BGN_AMT}입니다."
    ],
    9: [
        "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "문의 확인 후 해당 번호로 문자 안내 드리겠습니다.",
        "해당 문의는 문자로 추가 안내드리겠습니다."
    ],
    10: [
        "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "해당 문의에 대해 자세한 확인 후 문자로 답변드리겠습니다.",
        "해당 문의는 문자로 추가 안내드리겠습니다."
    ],
    11: [
        "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "매장에 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
        "문의 확인 후 전화드리겠습니다.",
        "해당 문의는 전화로 추가 안내드리겠습니다."
    ],
    12: [
        "저희 {STORE_NM}는 {ROAD_NM_ADDR}에 위치해 있습니다.",
        "{STORE_NM}의 위치는 {ROAD_NM_ADDR}입니다.",
        "{ROAD_NM_ADDR} 주소로 오시면 {STORE_NM} 매장을 찾으실 수 있습니다.",
        "{STORE_NM}의 정확한 주소는 {ROAD_NM_ADDR}입니다."
    ],
    13: [
        "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "해당 문의는 매장에 문의 부탁드립니다.",
        "해당 내용은 문의 확인 후 문자 안내드리겠습니다.",
        "해당 정보는 전화번호로 문자 안내드리겠습니다."
    ],
    14: [
        "해당 문의는 확인 후 전화하신 번호로 문자안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "매장에 문의 부탁드립니다.",
        "해당 내용은 문의 확인 후 문자 안내드리겠습니다.",
        "해당 정보는 전화번호로 문자 안내드리겠습니다."
    ],
    15: [
        "포인트 서비스나 쿠폰을 발급 받기 위해서는 회원 가입이 필요합니다. 회원가입은 스마트 주문 앱 또는 키오스크에서 가입 가능해요.",
        "회원 가입 후 포인트와 쿠폰을 받을 수 있습니다. 스마트 주문 앱을 이용해주세요.",
        "쿠폰과 포인트는 회원 가입 시 제공됩니다. 키오스크에서 가입 가능합니다.",
        "스마트 주문 앱이나 키오스크로 간편하게 회원 가입 후 혜택을 받아보세요.",
        "회원 가입 후 포인트 적립과 쿠폰 발급이 가능합니다."
    ],
    16: [
        "저희 {STORE_NM} 에서는 포인트 {USE_POSBL_BASE_POINT}점 부터 사용 가능해요.",
        "{STORE_NM}의 포인트 사용 기준은 {USE_POSBL_BASE_POINT}점입니다.",
        "{USE_POSBL_BASE_POINT}점 이상부터 포인트 사용 가능합니다.",
        "{STORE_NM}에서 포인트는 {USE_POSBL_BASE_POINT}점부터 이용 가능합니다.",
        "{USE_POSBL_BASE_POINT}점 이상 적립 후 포인트를 사용하실 수 있습니다."
    ],
    17: [
        "저희 {STORE_NM} 에서는 스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 발급해 주고 있어요.",
        "{STORE_NM}은 스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 제공합니다.",
        "{STAMP_TMS}개의 스탬프를 모으면 {COUPON_PACK_NM} 쿠폰을 받을 수 있습니다.",
        "스탬프 적립 {STAMP_TMS}개 달성 시 {COUPON_PACK_NM} 쿠폰이 발급됩니다.",
        "{STORE_NM}에서 스탬프 {STAMP_TMS}개 적립을 통해 쿠폰 혜택을 누릴 수 있습니다."
    ],
    18: [
        "현재 저희 {STORE_NM} 에서는 {EVENT_NM} 이벤트가 진행되고 있으며 {EVENT_BNEF_CND_CD} 동안 진행돼요.",
        "{STORE_NM}의 {EVENT_NM} 이벤트는 {EVENT_BNEF_CND_CD}까지입니다.",
        "{EVENT_NM}은 {EVENT_BNEF_CND_CD}까지 {STORE_NM}에서 참여 가능합니다.",
        "저희 {STORE_NM}에서 {EVENT_NM} 이벤트를 진행 중이며, {EVENT_BNEF_CND_CD}까지 진행됩니다.",
        "{EVENT_BNEF_CND_CD}까지 {EVENT_NM} 이벤트를 즐겨보세요."
    ],
    19: [
        "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
        "문의 확인 후 문자 안내 드리겠습니다.",
        "해당 문의는 문자로 추가 안내드리겠습니다."
    ],
    20: [
        "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
        "문의 내용을 확인하고 문자로 답변드리겠습니다.",
        "매장에 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
        "문의 확인 후 문자 드리겠습니다.",
        "해당 문의는 문자로 추가 안내드리겠습니다."
    ]
}

# 데이터를 표 형식으로 변환
data = []
for intent_num, intent_name in intents_dict.items():
    responses = response_templates.get(intent_num, ["응대 템플릿이 존재하지 않습니다."])
    for response in responses:
        data.append([intent_num, intent_name, response])

# DataFrame 생성
df = pd.DataFrame(data, columns=["Intent_num", "Intent", "User"])

import ace_tools as tools; tools.display_dataframe_to_user("Intent and Response Table", df)