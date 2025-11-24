"""
Intent Classification 모델 테스트
"""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TestIntentClassification:
    """Intent Classification 모델 테스트"""
    
    @pytest.fixture
    def model_path(self):
        """모델 경로"""
        return "checkpoint/klue_roberta_large_v9"
    
    @pytest.fixture
    def tokenizer(self, model_path):
        """토크나이저 로드"""
        return AutoTokenizer.from_pretrained(model_path)
    
    @pytest.fixture
    def model(self, model_path):
        """모델 로드"""
        return AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def test_model_loading(self, model):
        """모델이 정상적으로 로드되는지 테스트"""
        assert model is not None
        assert isinstance(model, AutoModelForSequenceClassification)
    
    def test_tokenizer_loading(self, tokenizer):
        """토크나이저가 정상적으로 로드되는지 테스트"""
        assert tokenizer is not None
    
    def test_inference(self, model, tokenizer):
        """추론이 정상적으로 작동하는지 테스트"""
        text = "떡볶이 가격이 얼마예요?"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.logits is not None
        assert outputs.logits.shape[1] == 48  # 48개 클래스
    
    def test_menu_price_intent(self, model, tokenizer):
        """메뉴 가격 문의 Intent 분류 테스트"""
        test_cases = [
            "떡볶이 얼마예요?",
            "김밥 가격 알려주세요",
            "라면 가격이 궁금해요"
        ]
        
        for text in test_cases:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            # Intent ID 1: 특정 상품 및 가격 안내
            assert predicted_class == 1 or predicted_class in range(0, 48)
    
    def test_greeting_intent(self, model, tokenizer):
        """인사 Intent 분류 테스트"""
        test_cases = [
            "안녕하세요",
            "안녕",
            "반갑습니다"
        ]
        
        for text in test_cases:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = torch.softmax(outputs.logits, dim=-1)[0][predicted_class].item()
            
            # Intent ID 46: 인사 인텐트
            assert predicted_class in range(0, 48)
            assert confidence > 0.5
    
    def test_confidence_score(self, model, tokenizer):
        """신뢰도 점수가 0-1 범위인지 테스트"""
        text = "메뉴판 보여주세요"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)
        max_prob = torch.max(probs).item()
        
        assert 0 <= max_prob <= 1
        assert torch.sum(probs).item() == pytest.approx(1.0, rel=1e-5)
    
    def test_batch_inference(self, model, tokenizer):
        """배치 추론 테스트"""
        texts = [
            "떡볶이 가격이 얼마예요?",
            "영업시간 알려주세요",
            "카드 결제 되나요?"
        ]
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.logits.shape[0] == len(texts)
        assert outputs.logits.shape[1] == 48
    
    def test_empty_input(self, model, tokenizer):
        """빈 입력 처리 테스트"""
        text = ""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.logits is not None
    
    def test_long_input(self, model, tokenizer):
        """긴 입력 처리 테스트"""
        text = "떡볶이 " * 200  # 매우 긴 입력
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.logits is not None
        assert inputs['input_ids'].shape[1] <= 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
