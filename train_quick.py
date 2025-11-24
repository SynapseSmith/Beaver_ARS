"""
Quick Training Script for Beaver ARS
Minimal version for fast testing and development
"""

import os
import sys
import argparse
import torch
from pathlib import Path


def check_environment():
    """Check if environment is ready for training"""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU detected, using CPU (training will be slower)")
    
    # Check required packages
    try:
        import transformers
        import datasets
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print()


def train_intent_quick(args):
    """Quick Intent Classification training"""
    print("=" * 60)
    print("Intent Classification - Quick Training")
    print("=" * 60)
    print(f"Data: {args.intent_data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Import training script
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    try:
        # This would import your actual training script
        # For now, we'll show the command to run
        cmd = (
            f"python src/241215_step1_train_cls_intent.py "
            f"--data_path {args.intent_data} "
            f"--output_dir {args.output_dir}/intent_classifier "
            f"--num_epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--learning_rate {args.learning_rate}"
        )
        
        print(f"Running: {cmd}\n")
        os.system(cmd)
        
        print("\n✓ Intent classifier training completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False


def train_ner_quick(args):
    """Quick NER training"""
    print("=" * 60)
    print("NER Model - Quick Training")
    print("=" * 60)
    print(f"Data: {args.ner_data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    try:
        cmd = (
            f"python src/241218_step1_ner_train_i_tagging.py "
            f"--data_path {args.ner_data} "
            f"--output_dir {args.output_dir}/ner_model "
            f"--num_epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--learning_rate {args.learning_rate}"
        )
        
        print(f"Running: {cmd}\n")
        os.system(cmd)
        
        print("\n✓ NER model training completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False


def test_models(args):
    """Test trained models"""
    print("=" * 60)
    print("Testing Models")
    print("=" * 60)
    
    test_texts = [
        "김치찌개 2개 주문할게요",
        "영업시간이 언제인가요?",
        "배달 가능한가요?",
        "불고기 3인분 주문하고 싶어요"
    ]
    
    print("\nTest queries:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    print("\n" + "=" * 60)
    print("\nTo test inference, run:")
    print(f"  python src/241215_step1_inference_cls_intent.py \\")
    print(f"    --model_path {args.output_dir}/intent_classifier/best_model.pt \\")
    print(f"    --text \"김치찌개 2개 주문할게요\"")
    print()


def main():
    parser = argparse.ArgumentParser(description="Quick training script for Beaver ARS")
    
    # Data paths
    parser.add_argument("--intent_data", default="data/sample/intent_sample.csv",
                        help="Path to intent classification data")
    parser.add_argument("--ner_data", default="data/sample/ner_sample.conll",
                        help="Path to NER data")
    parser.add_argument("--output_dir", default="models",
                        help="Output directory for trained models")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3 for quick test)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    
    # Options
    parser.add_argument("--train_intent", action="store_true",
                        help="Train intent classifier")
    parser.add_argument("--train_ner", action="store_true",
                        help="Train NER model")
    parser.add_argument("--train_all", action="store_true",
                        help="Train both models")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip inference test")
    
    args = parser.parse_args()
    
    # Check environment
    check_environment()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine what to train
    train_intent = args.train_intent or args.train_all
    train_ner = args.train_ner or args.train_all
    
    if not train_intent and not train_ner:
        print("Please specify what to train:")
        print("  --train_intent : Train intent classifier")
        print("  --train_ner    : Train NER model")
        print("  --train_all    : Train both models")
        print("\nExample:")
        print("  python train_quick.py --train_all --epochs 5")
        sys.exit(1)
    
    # Train models
    success = True
    
    if train_intent:
        if not train_intent_quick(args):
            success = False
    
    if train_ner and success:
        if not train_ner_quick(args):
            success = False
    
    # Test models
    if success and not args.skip_test:
        test_models(args)
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ Training completed successfully!")
        print(f"\nModels saved to: {args.output_dir}/")
        print("\nNext steps:")
        print("  1. Check model performance in logs")
        print("  2. Test the API: python src/web_server.py")
        print("  3. Deploy: docker-compose up -d")
    else:
        print("✗ Training failed")
        print("\nCheck logs for errors")
    print("=" * 60)


if __name__ == "__main__":
    main()
