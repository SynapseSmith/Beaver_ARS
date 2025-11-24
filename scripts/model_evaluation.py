"""
모델 평가 스크립트

학습된 모델의 성능을 종합적으로 평가합니다.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None
    ):
        """초기화"""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else model_path
        )
        self.model.eval()
        
        # GPU 사용 가능 시
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """예측 수행"""
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'intent_num'
    ) -> Dict:
        """종합 평가"""
        texts = test_df[text_column].tolist()
        true_labels = test_df[label_column].values
        
        # 예측
        predictions, probabilities = self.predict(texts)
        
        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_macro': f1_score(true_labels, predictions, average='macro'),
            'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
            'precision_macro': precision_score(true_labels, predictions, average='macro'),
            'recall_macro': recall_score(true_labels, predictions, average='macro'),
        }
        
        # Confidence 분석
        max_probs = np.max(probabilities, axis=1)
        metrics['avg_confidence'] = np.mean(max_probs)
        metrics['min_confidence'] = np.min(max_probs)
        metrics['max_confidence'] = np.max(max_probs)
        
        return metrics, predictions, probabilities
    
    def plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        class_names: List[str] = None,
        save_path: str = 'confusion_matrix.png'
    ):
        """Confusion Matrix 시각화"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    
    def plot_confidence_distribution(
        self,
        probabilities: np.ndarray,
        save_path: str = 'confidence_dist.png'
    ):
        """Confidence 분포 시각화"""
        max_probs = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=50, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.axvline(np.mean(max_probs), color='r', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved confidence distribution to {save_path}")
    
    def analyze_errors(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        top_k: int = 10
    ) -> pd.DataFrame:
        """오류 분석"""
        test_df = test_df.copy()
        test_df['prediction'] = predictions
        test_df['confidence'] = np.max(probabilities, axis=1)
        test_df['correct'] = test_df['intent_num'] == test_df['prediction']
        
        # 오분류 케이스
        errors = test_df[~test_df['correct']].copy()
        errors = errors.sort_values('confidence', ascending=False)
        
        print(f"\nTotal errors: {len(errors)} ({len(errors)/len(test_df)*100:.2f}%)")
        print(f"\nTop {top_k} confident errors:")
        print(errors[['text', 'intent', 'intent_num', 'prediction', 'confidence']].head(top_k))
        
        return errors


def main():
    """메인 함수"""
    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    
    # 평가자 초기화
    evaluator = ModelEvaluator(
        model_path='checkpoint/klue_roberta_large_v9'
    )
    
    # 테스트 데이터 로드
    test_df = pd.read_csv('241215_BERT/data/test.csv')
    
    # 평가
    print("\nEvaluating...")
    metrics, predictions, probabilities = evaluator.evaluate(test_df)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # Intent 이름 로드
    intent_dict = {
        0: "메뉴 카테고리 안내",
        1: "특정 상품 및 가격 안내",
        # ... (전체 48개)
    }
    
    # Confusion Matrix
    evaluator.plot_confusion_matrix(
        test_df['intent_num'].values,
        predictions,
        class_names=[intent_dict.get(i, f"Intent {i}") for i in range(48)]
    )
    
    # Confidence 분포
    evaluator.plot_confidence_distribution(probabilities)
    
    # 오류 분석
    errors = evaluator.analyze_errors(test_df, predictions, probabilities)
    errors.to_csv('error_analysis.csv', index=False)
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
