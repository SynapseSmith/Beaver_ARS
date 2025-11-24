"""
모델 내보내기 스크립트

학습된 모델을 다양한 형식으로 내보냅니다.
- PyTorch (.pth)
- ONNX (.onnx)
- TorchScript (.pt)
"""

import torch
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


class ModelExporter:
    """모델 내보내기 클래스"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = 'exported_models'
    ):
        """초기화"""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # 출력 디렉토리 생성
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_pytorch(self, filename: str = 'model.pth'):
        """PyTorch 형식으로 내보내기"""
        output_path = self.output_dir / filename
        
        # 모델 state_dict 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config.to_dict()
        }, output_path)
        
        # 토크나이저 저장
        self.tokenizer.save_pretrained(self.output_dir / 'tokenizer')
        
        print(f"✅ Exported PyTorch model to {output_path}")
        return output_path
    
    def export_onnx(self, filename: str = 'model.onnx'):
        """ONNX 형식으로 내보내기"""
        output_path = self.output_dir / filename
        
        # 더미 입력 생성
        dummy_text = "테스트 문장입니다"
        inputs = self.tokenizer(
            dummy_text,
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True
        )
        
        # ONNX 변환
        torch.onnx.export(
            self.model,
            (inputs['input_ids'], inputs['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch'}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # ONNX 모델 검증
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ Exported ONNX model to {output_path}")
        return output_path
    
    def export_torchscript(self, filename: str = 'model.pt'):
        """TorchScript 형식으로 내보내기"""
        output_path = self.output_dir / filename
        
        # 더미 입력
        dummy_text = "테스트 문장입니다"
        inputs = self.tokenizer(
            dummy_text,
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True
        )
        
        # TorchScript 변환
        traced_model = torch.jit.trace(
            self.model,
            (inputs['input_ids'], inputs['attention_mask'])
        )
        
        # 저장
        traced_model.save(str(output_path))
        
        print(f"✅ Exported TorchScript model to {output_path}")
        return output_path
    
    def optimize_onnx(self, onnx_path: str):
        """ONNX 모델 최적화"""
        from onnxruntime.transformers import optimizer
        
        optimized_path = str(self.output_dir / 'model_optimized.onnx')
        
        # 최적화
        optimizer.optimize_model(
            str(onnx_path),
            model_type='bert',
            num_heads=12,  # RoBERTa-large
            hidden_size=1024,
            optimization_options=None
        )
        
        print(f"✅ Optimized ONNX model saved to {optimized_path}")
        return optimized_path
    
    def quantize_model(self):
        """모델 양자화 (INT8)"""
        from torch.quantization import quantize_dynamic
        
        # Dynamic Quantization
        quantized_model = quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 저장
        output_path = self.output_dir / 'model_quantized.pth'
        torch.save(quantized_model.state_dict(), output_path)
        
        print(f"✅ Quantized model saved to {output_path}")
        return output_path
    
    def test_exported_model(self, model_path: str, format: str = 'onnx'):
        """내보낸 모델 테스트"""
        test_text = "떡볶이 가격이 얼마예요?"
        
        if format == 'onnx':
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(model_path))
            
            inputs = self.tokenizer(
                test_text,
                return_tensors='np',
                padding='max_length',
                max_length=128,
                truncation=True
            )
            
            outputs = session.run(
                None,
                {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
            )
            
            logits = outputs[0]
            predicted_class = logits.argmax(axis=-1)[0]
            
            print(f"\n🧪 Test inference:")
            print(f"  Input: {test_text}")
            print(f"  Predicted class: {predicted_class}")
        
        elif format == 'torchscript':
            model = torch.jit.load(str(model_path))
            
            inputs = self.tokenizer(
                test_text,
                return_tensors='pt',
                padding='max_length',
                max_length=128,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
            
            predicted_class = outputs.logits.argmax(dim=-1).item()
            
            print(f"\n🧪 Test inference:")
            print(f"  Input: {test_text}")
            print(f"  Predicted class: {predicted_class}")


def main():
    """메인 함수"""
    print("=" * 50)
    print("Model Export")
    print("=" * 50)
    
    # Exporter 초기화
    exporter = ModelExporter(
        model_path='checkpoint/klue_roberta_large_v9',
        output_dir='exported_models'
    )
    
    # PyTorch 내보내기
    print("\n1. Exporting PyTorch model...")
    exporter.export_pytorch()
    
    # ONNX 내보내기
    print("\n2. Exporting ONNX model...")
    onnx_path = exporter.export_onnx()
    
    # ONNX 테스트
    print("\n3. Testing ONNX model...")
    exporter.test_exported_model(onnx_path, format='onnx')
    
    # TorchScript 내보내기
    print("\n4. Exporting TorchScript model...")
    ts_path = exporter.export_torchscript()
    
    # TorchScript 테스트
    print("\n5. Testing TorchScript model...")
    exporter.test_exported_model(ts_path, format='torchscript')
    
    # 양자화 (선택)
    print("\n6. Quantizing model...")
    exporter.quantize_model()
    
    print("\n✅ All exports completed successfully!")


if __name__ == "__main__":
    main()
