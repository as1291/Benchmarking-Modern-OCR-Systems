import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer
)
from PIL import Image, ImageEnhance, ImageOps
import os
import numpy as np
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import huggingface_hub

class OCRDataset(Dataset):
    def __init__(self, image_paths, texts, processor, max_length=128, augment=False):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        
        # Define augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentations if enabled
        if self.augment:
            # Convert PIL image to numpy array for albumentations
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = Image.fromarray(augmented['image'].numpy().transpose(1, 2, 0))
        
        # Process image and text
        encoding = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
            
        return encoding

class ModelFinetuner:
    def __init__(self, model_name, device=None, hf_token=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set Hugging Face token if provided
        if hf_token:
            huggingface_hub.login(token=hf_token)
            print("Logged in to Hugging Face")
        
        # Initialize model and processor
        if model_name == "qwen":
            try:
                # Use the correct model name
                model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
                print(f"Loading model: {model_id}")
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForVision2Seq.from_pretrained(model_id)
            except Exception as e:
                print(f"Error loading Qwen model: {e}")
                print("Falling back to TrOCR")
                model_id = "microsoft/trocr-base-printed"
                self.processor = TrOCRProcessor.from_pretrained(model_id)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        else:  # florence
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        
        self.model.to(self.device)

    def prepare_dataset(self, image_paths, texts, augment=False):
        """Prepare dataset for fine-tuning"""
        return OCRDataset(image_paths, texts, self.processor, augment=augment)

    def finetune(self, train_dataset, eval_dataset=None, output_dir="./finetuned_model", 
                 num_train_epochs=3, batch_size=8, learning_rate=5e-5, warmup_steps=100):
        """Fine-tune the model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            push_to_hub=False,
            warmup_steps=warmup_steps,
            logging_steps=10,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
            gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def evaluate(self, test_dataset):
        """Evaluate the fine-tuned model"""
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./eval_results"),
        )
        
        metrics = trainer.evaluate(test_dataset)
        return metrics

def prepare_training_data(image_data):
    """Prepare training data from the image dataset"""
    image_paths = []
    texts = []
    
    for item in image_data:
        image_paths.append(item["path"])
        texts.append(item["text"])
    
    return image_paths, texts

if __name__ == "__main__":
    # Example usage
    from ocrcode import download_test_images
    
    # Get Hugging Face token from environment variable or prompt user
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token (or press Enter to skip): ")
        if not hf_token:
            hf_token = None
    
    # Download test images
    image_data = download_test_images()
    
    # Prepare training data
    image_paths, texts = prepare_training_data(image_data)
    
    # Split data into train and eval sets (80-20 split)
    train_size = int(0.8 * len(image_paths))
    train_image_paths = image_paths[:train_size]
    train_texts = texts[:train_size]
    eval_image_paths = image_paths[train_size:]
    eval_texts = texts[train_size:]
    
    # Fine-tune Qwen model
    print("Fine-tuning Qwen model...")
    qwen_finetuner = ModelFinetuner("qwen", hf_token=hf_token)
    train_dataset = qwen_finetuner.prepare_dataset(train_image_paths, train_texts, augment=True)
    eval_dataset = qwen_finetuner.prepare_dataset(eval_image_paths, eval_texts, augment=False)
    qwen_finetuner.finetune(
        train_dataset, 
        eval_dataset, 
        output_dir="./finetuned_qwen",
        num_train_epochs=5,
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100
    )
    
    # Fine-tune Florence model
    print("Fine-tuning Florence model...")
    florence_finetuner = ModelFinetuner("florence", hf_token=hf_token)
    train_dataset = florence_finetuner.prepare_dataset(train_image_paths, train_texts, augment=True)
    eval_dataset = florence_finetuner.prepare_dataset(eval_image_paths, eval_texts, augment=False)
    florence_finetuner.finetune(
        train_dataset,
        eval_dataset,
        output_dir="./finetuned_florence",
        num_train_epochs=5,
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100
    ) 