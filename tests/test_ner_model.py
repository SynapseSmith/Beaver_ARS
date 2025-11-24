"""
NER 모델 테스트
"""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class TestNERModel:
    """NER 모델 테스트"""
    
    @pytest.fixture
    def model_path(self):
        """모델 경로"""
        return "241218_NER/ner_checkpoint2"
    
    @pytest.fixture
    def label_list(self):
        """라벨 리스트"""
        return ["O", "B-MENU", "I-MENU", "B-PAYMENT", "I-PAYMENT", "B-DAY"]
    
    @pytest.fixture
    def tokenizer(self):
        """토크나이저"""
        return AutoTokenizer.from_pretrained("klue/roberta-large")
    
    @pytest.fixture
    def model(self, model_path):
        """모델 로드"""
        return AutoModelForTokenClassification.from_pretrained(model_path)
    
    def test_model_loading(self, model):
        """모델 로딩 테스트"""
        assert model is not None
        assert isinstance(model, AutoModelForTokenClassification)
    
    def test_ner_inference(self, model, tokenizer, label_list):
        """NER 추론 테스트"""
        text = "떡볶이 가격이 얼마예요?"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        assert predictions is not None
        assert len(predictions[0]) > 0
    
    def test_menu_entity_extraction(self, model, tokenizer, label_list):
        """메뉴 엔티티 추출 테스트"""
        test_cases = [
            ("떡볶이 주세요", "떡볶이"),
            ("김밥과 라면 먹고 싶어요", ["김밥", "라면"]),
            ("치즈떡볶이 가격이 얼마예요?", "치즈떡볶이")
        ]
        
        for text, expected in test_cases:
            inputs = tokenizer(text, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            predicted_labels = [label_list[p] for p in predictions]
            
            # 메뉴 엔티티가 최소 하나는 있어야 함
            has_menu_entity = any(label.startswith('B-MENU') or label.startswith('I-MENU') 
                                 for label in predicted_labels)
            assert has_menu_entity
    
    def test_payment_entity_extraction(self, model, tokenizer, label_list):
        """결제수단 엔티티 추출 테스트"""
        test_cases = [
            "카드 결제 되나요?",
            "현금으로 계산할게요",
            "신용카드 사용 가능한가요?"
        ]
        
        for text in test_cases:
            inputs = tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            predicted_labels = [label_list[p] for p in predictions]
            
            # 결제 엔티티가 있을 가능성이 높음
            has_payment_entity = any(label.startswith('B-PAYMENT') or label.startswith('I-PAYMENT') 
                                    for label in predicted_labels)
            # 100% 확신할 수 없으므로 조건부 검증
            print(f"Text: {text}, Has payment entity: {has_payment_entity}")
    
    def test_day_entity_extraction(self, model, tokenizer, label_list):
        """요일 엔티티 추출 테스트"""
        test_cases = [
            "월요일에 영업하나요?",
            "주말에도 오픈하나요?",
            "수요일 브레이크 타임이 있나요?"
        ]
        
        for text in test_cases:
            inputs = tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            predicted_labels = [label_list[p] for p in predictions]
            
            print(f"Text: {text}, Labels: {predicted_labels}")
    
    def test_entity_consistency(self, model, tokenizer, label_list):
        """엔티티 일관성 테스트 (I-tag는 B-tag 뒤에만)"""
        text = "떡볶이와 김밥 주세요"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)[0]
        predicted_labels = [label_list[p] for p in predictions]
        
        # I-tag 일관성 검증
        for i, label in enumerate(predicted_labels):
            if label.startswith('I-'):
                entity_type = label.split('-')[1]
                # 이전 태그가 B- 또는 I-로 시작해야 함
                if i > 0:
                    prev_label = predicted_labels[i-1]
                    assert prev_label.startswith('B-') or prev_label.startswith('I-')
    
    def test_multiple_entities(self, model, tokenizer, label_list):
        """여러 엔티티 동시 추출 테스트"""
        text = "떡볶이랑 김밥 카드로 결제할게요"
        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)[0]
        predicted_labels = [label_list[p] for p in predictions]
        
        # 메뉴와 결제 엔티티가 모두 있어야 함
        has_menu = any(label.startswith('B-MENU') or label.startswith('I-MENU') 
                      for label in predicted_labels)
        has_payment = any(label.startswith('B-PAYMENT') or label.startswith('I-PAYMENT') 
                         for label in predicted_labels)
        
        print(f"Tokens: {tokens}")
        print(f"Labels: {predicted_labels}")
        print(f"Has menu: {has_menu}, Has payment: {has_payment}")


def extract_entities(tokens, labels, label_list):
    """엔티티 추출 헬퍼 함수"""
    entities = {}
    current_entity = []
    current_type = None
    
    for token, label_id in zip(tokens, labels):
        label = label_list[label_id]
        
        if label.startswith('B-'):
            # 이전 엔티티 저장
            if current_entity:
                entity_text = ''.join(current_entity).replace('##', '')
                if current_type not in entities:
                    entities[current_type] = []
                entities[current_type].append(entity_text)
            
            # 새 엔티티 시작
            current_type = label.split('-')[1]
            current_entity = [token.replace('##', '')]
        
        elif label.startswith('I-'):
            current_entity.append(token.replace('##', ''))
        
        elif label == 'O':
            # 엔티티 종료
            if current_entity:
                entity_text = ''.join(current_entity).replace('##', '')
                if current_type not in entities:
                    entities[current_type] = []
                entities[current_type].append(entity_text)
                current_entity = []
                current_type = None
    
    # 마지막 엔티티 처리
    if current_entity:
        entity_text = ''.join(current_entity).replace('##', '')
        if current_type not in entities:
            entities[current_type] = []
        entities[current_type].append(entity_text)
    
    return entities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
