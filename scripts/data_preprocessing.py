"""
데이터 전처리 스크립트

Intent 데이터와 NER 데이터를 정제하고 전처리합니다.
"""

import pandas as pd
import re
from typing import List, Tuple


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정제"""
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        # 특수문자 정규화 (선택적)
        # text = re.sub(r'[^\w\s가-힣]', '', text)
        
        return text
    
    @staticmethod
    def normalize_intent_data(
        csv_path: str,
        output_path: str
    ) -> pd.DataFrame:
        """Intent 데이터 정규화"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 텍스트 정제
        df['text'] = df['text'].apply(DataPreprocessor.clean_text)
        
        # 중복 제거
        original_len = len(df)
        df = df.drop_duplicates(subset=['text'])
        print(f"Removed {original_len - len(df)} duplicates")
        
        # 빈 텍스트 제거
        df = df[df['text'].str.len() > 0]
        
        # 라벨 확인
        print(f"\nIntent distribution:")
        print(df['intent'].value_counts())
        
        # 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved to {output_path}")
        
        return df
    
    @staticmethod
    def validate_conll_data(
        conll_path: str
    ) -> Tuple[int, int, List[str]]:
        """CoNLL 데이터 검증"""
        sentences = []
        current_sentence = []
        errors = []
        
        with open(conll_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            parts = line.split('\t')
            
            if len(parts) != 2:
                errors.append(f"Line {i}: Invalid format (expected 2 columns)")
                continue
            
            token, label = parts
            
            # 라벨 검증
            valid_labels = ['O', 'B-MENU', 'I-MENU', 'B-PAYMENT', 'I-PAYMENT', 'B-DAY']
            if label not in valid_labels:
                errors.append(f"Line {i}: Invalid label '{label}'")
            
            # I-tag 일관성 검증
            if label.startswith('I-'):
                if not current_sentence:
                    errors.append(f"Line {i}: I-tag at sentence start")
                else:
                    prev_label = current_sentence[-1][1]
                    entity_type = label.split('-')[1]
                    if not (prev_label.startswith('B-') or prev_label.startswith('I-')):
                        errors.append(f"Line {i}: I-tag without preceding B-tag or I-tag")
            
            current_sentence.append((token, label))
        
        if current_sentence:
            sentences.append(current_sentence)
        
        print(f"Total sentences: {len(sentences)}")
        print(f"Total tokens: {sum(len(s) for s in sentences)}")
        
        if errors:
            print(f"\n❌ Found {len(errors)} errors:")
            for error in errors[:10]:  # 처음 10개만 출력
                print(f"  - {error}")
        else:
            print("\n✅ No errors found")
        
        return len(sentences), sum(len(s) for s in sentences), errors
    
    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터셋을 train/test로 분할"""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['intent_num']  # 클래스 비율 유지
        )
        
        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")
        
        return train_df, test_df
    
    @staticmethod
    def augment_data(
        texts: List[str],
        method: str = 'synonym'
    ) -> List[str]:
        """데이터 증강"""
        augmented = []
        
        if method == 'synonym':
            # 동의어 치환 (간단한 예시)
            synonym_dict = {
                '얼마': ['얼마나', '가격이 어떻게'],
                '주세요': ['주시겠어요', '부탁드려요'],
                '가능한가요': ['되나요', '할 수 있나요'],
            }
            
            for text in texts:
                for original, synonyms in synonym_dict.items():
                    if original in text:
                        for syn in synonyms:
                            augmented.append(text.replace(original, syn))
        
        return augmented


def main():
    """메인 함수"""
    preprocessor = DataPreprocessor()
    
    # Intent 데이터 전처리
    print("=" * 50)
    print("Intent Data Preprocessing")
    print("=" * 50)
    
    intent_df = preprocessor.normalize_intent_data(
        csv_path='241215_BERT/data/user_intent_v4.csv',
        output_path='241215_BERT/data/user_intent_v4_clean.csv'
    )
    
    # Train/Test 분할
    train_df, test_df = preprocessor.split_dataset(intent_df)
    train_df.to_csv('241215_BERT/data/train.csv', index=False)
    test_df.to_csv('241215_BERT/data/test.csv', index=False)
    
    # NER 데이터 검증
    print("\n" + "=" * 50)
    print("NER Data Validation")
    print("=" * 50)
    
    preprocessor.validate_conll_data(
        conll_path='241218_NER/data/NER_labeled_data_v2.conll'
    )


if __name__ == "__main__":
    main()
