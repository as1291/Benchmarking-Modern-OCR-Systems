import os
import time
import huggingface_hub
from model_finetuning import ModelFinetuner, prepare_training_data
from ocrcode import download_test_images, OCRModelExtractor

def get_hf_token():
    """Get Hugging Face token from environment variable or prompt user"""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token (or press Enter to skip): ")
        if not hf_token:
            hf_token = None
    return hf_token

def pretrain_models():
    """Pre-train the Qwen and Florence models on the test data"""
    print("Starting pre-training process...")
    
    # Get Hugging Face token
    hf_token = get_hf_token()
    if hf_token:
        huggingface_hub.login(token=hf_token)
        print("Logged in to Hugging Face")
    
    # Download test images
    print("Downloading test images...")
    image_data = download_test_images()
    
    # Prepare training data
    print("Preparing training data...")
    image_paths, texts = prepare_training_data(image_data)
    
    # Split data into train and eval sets (80-20 split)
    train_size = int(0.8 * len(image_paths))
    train_image_paths = image_paths[:train_size]
    train_texts = texts[:train_size]
    eval_image_paths = image_paths[train_size:]
    eval_texts = texts[train_size:]
    
    # Fine-tune Qwen model
    print("\nFine-tuning Qwen model...")
    qwen_finetuner = ModelFinetuner("qwen", hf_token=hf_token)
    train_dataset = qwen_finetuner.prepare_dataset(train_image_paths, train_texts, augment=True)
    eval_dataset = qwen_finetuner.prepare_dataset(eval_image_paths, eval_texts, augment=False)
    qwen_finetuner.finetune(
        train_dataset, 
        eval_dataset, 
        output_dir="./finetuned_qwen",
        num_train_epochs=5,  # Increased epochs for better training
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100  # Added warmup steps for better convergence
    )
    
    # Fine-tune Florence model
    print("\nFine-tuning Florence model...")
    florence_finetuner = ModelFinetuner("florence", hf_token=hf_token)
    train_dataset = florence_finetuner.prepare_dataset(train_image_paths, train_texts, augment=True)
    eval_dataset = florence_finetuner.prepare_dataset(eval_image_paths, eval_texts, augment=False)
    florence_finetuner.finetune(
        train_dataset,
        eval_dataset,
        output_dir="./finetuned_florence",
        num_train_epochs=5,  # Increased epochs for better training
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100  # Added warmup steps for better convergence
    )
    
    print("\nPre-training completed successfully!")
    return True

def run_tests():
    """Run tests using the pre-trained models"""
    print("\nStarting test evaluation...")
    
    # Get Hugging Face token
    hf_token = get_hf_token()
    if hf_token:
        # Set environment variable for ocrcode.py to use
        os.environ["HF_TOKEN"] = hf_token
    
    # Download test images if not already done
    image_data = download_test_images()
    
    # Initialize OCR model extractor (will use pre-trained models if available)
    ocr_extractor = OCRModelExtractor()
    
    # Setup all OCR models
    num_models_loaded = ocr_extractor.setup_all_models()
    print(f"\nSuccessfully loaded {num_models_loaded} OCR models.")
    
    if ocr_extractor.models:
        # Evaluate all test images
        print("\nEvaluating OCR models on test images...")
        evaluation_results = ocr_extractor.evaluate_all_images(image_data)
        
        # Calculate and print overall metrics
        print("\nOverall Evaluation Metrics:")
        overall_metrics = ocr_extractor.calculate_overall_metrics()
        for model, metrics in overall_metrics.items():
            print(f"\n--- {model} ---")
            print(f"  Average Character Accuracy: {metrics.get('avg_char_accuracy', 'N/A'):.4f}")
            print(f"  Average Levenshtein Distance: {metrics.get('avg_levenshtein', 'N/A'):.4f}")
            print(f"  Average Exact Match Ratio: {metrics.get('avg_exact_match', 'N/A'):.4f}")
            print(f"  Average Processing Time: {metrics['avg_processing_time']:.4f} seconds")
        
        # Plot comparison graphs
        print("\nGenerating comparison graphs...")
        ocr_extractor.plot_comparison_graphs()
        print("Comparison graphs saved to the 'results' directory.")
    else:
        print("No OCR models were loaded. Please check the setup process.")

if __name__ == "__main__":
    # Check if models are already pre-trained
    if os.path.exists("./finetuned_qwen") and os.path.exists("./finetuned_florence"):
        print("Pre-trained models already exist. Skipping pre-training.")
        run_tests()
    else:
        # Pre-train models first
        if pretrain_models():
            # Then run tests
            run_tests()
        else:
            print("Pre-training failed. Please check the errors above.") 