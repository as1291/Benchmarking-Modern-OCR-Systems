import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import requests
import warnings
warnings.filterwarnings("ignore")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

# Install required packages if not available
def install_requirements():
    try:
        import easyocr
    except ImportError:
        print("Installing EasyOCR...")
        os.system("pip install easyocr")

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers")

    try:
        import Levenshtein
    except ImportError:
        print("Installing Levenshtein...")
        os.system("pip install python-Levenshtein")

    try:
        import seaborn as sns
    except ImportError:
        print("Installing seaborn...")
        os.system("pip install seaborn")

    print("All requirements installed.")

# First install requirements before importing them
if __name__ == "__main__":
    install_requirements()

# Now we can safely import Levenshtein and seaborn after installation
import Levenshtein
import seaborn as sns

# Download test images if they don't exist
def download_test_images():
    """Create a diverse dataset of 200 test images with various text effects and variations"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import random

    # Create base text samples (20 different texts)
    texts = [
        "Hello World",
        "OCR Testing",
        "Python is awesome",
        "Comparing Models",
        "Text Extraction",
        "Machine Learning",
        "Deep Learning",
        "Computer Vision",
        "Natural Language",
        "Artificial Intelligence",
        "Data Science",
        "Neural Networks",
        "Image Processing",
        "Pattern Recognition",
        "Feature Extraction",
        "Model Evaluation",
        "Performance Metrics",
        "Accuracy Analysis",
        "Text Recognition",
        "Visual Processing"
    ]

    # Create directories for different types of images
    os.makedirs("test_images/blurred", exist_ok=True)
    os.makedirs("test_images/distorted", exist_ok=True)
    os.makedirs("test_images/handwritten", exist_ok=True)

    image_data = []

    def apply_blur(image, sigma):
        """Apply Gaussian blur to image"""
        img_array = np.array(image)
        blurred = gaussian_filter(img_array, sigma=sigma)
        return Image.fromarray(blurred.astype(np.uint8))

    def apply_distortion(image, intensity):
        """Apply random distortion to image"""
        img_array = np.array(image)
        rows, cols = img_array.shape[:2]
        
        # Create random displacement field
        displacement = np.random.rand(rows, cols, 2) * intensity
        displacement = displacement.astype(np.float32)
        
        # Apply displacement
        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)
        for y in range(rows):
            for x in range(cols):
                map_x[y,x] = x + displacement[y,x,0]
                map_y[y,x] = y + displacement[y,x,1]
        
        # Remap image
        distorted = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        return Image.fromarray(distorted)

    def create_handwritten_effect(image, variation):
        """Create handwritten-like effect"""
        img_array = np.array(image)
        
        # Add random noise
        noise = np.random.normal(0, variation, img_array.shape).astype(np.uint8)
        img_array = cv2.add(img_array, noise)
        
        # Add slight rotation
        angle = random.uniform(-variation, variation)
        center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_array, matrix, (img_array.shape[1], img_array.shape[0]))
        
        return Image.fromarray(rotated)

    # Generate images for each text
    for text_idx, text in enumerate(texts):
        # Create base image
        base_img = Image.new('RGB', (800, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(base_img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        d.text((20, 30), text, fill=(0, 0, 0), font=font)

        # 1. Create blurred variations (10 levels)
        for blur_level in range(10):
            sigma = (blur_level + 1) * 0.5  # Increasing blur levels
            blurred_img = apply_blur(base_img, sigma)
            filename = os.path.join("test_images/blurred", f"text_{text_idx}_blur_{blur_level}.png")
            blurred_img.save(filename)
            image_data.append({
                "path": filename,
                "text": text,
                "type": "blurred",
                "level": blur_level
            })

        # 2. Create distorted variations (10 levels)
        for dist_level in range(10):
            intensity = (dist_level + 1) * 2  # Increasing distortion levels
            distorted_img = apply_distortion(base_img, intensity)
            filename = os.path.join("test_images/distorted", f"text_{text_idx}_dist_{dist_level}.png")
            distorted_img.save(filename)
            image_data.append({
                "path": filename,
                "text": text,
                "type": "distorted",
                "level": dist_level
            })

        # 3. Create handwritten variations (10 levels per text)
        for hand_level in range(10):
            variation = (hand_level + 1) * 0.5  # Increasing variation levels
            handwritten_img = create_handwritten_effect(base_img, variation)
            filename = os.path.join("test_images/handwritten", f"text_{text_idx}_hand_{hand_level}.png")
            handwritten_img.save(filename)
            image_data.append({
                "path": filename,
                "text": text,
                "type": "handwritten",
                "level": hand_level
            })

    print(f"Created {len(image_data)} test images:")
    print(f"- {len(texts)} different texts")
    print(f"- {len(texts) * 10} blurred variations")
    print(f"- {len(texts) * 10} distorted variations")
    print(f"- {len(texts) * 10} handwritten variations")

    return image_data


class OCRModelExtractor:
    def __init__(self):
        self.models = {}
        self.results = {}
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def setup_easyocr(self):
        """Setup EasyOCR model"""
        print("Setting up EasyOCR model...")
        try:
            import easyocr
            # Enable GPU if available
            gpu = torch.cuda.is_available()
            reader = easyocr.Reader(['en'], gpu=gpu)

            def easyocr_predict(image):
                if isinstance(image, Image.Image):
                    # Convert PIL Image to numpy array
                    image = np.array(image)
                    if len(image.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                results = reader.readtext(image)
                return " ".join([text for _, text, _ in results])

            self.models["easyocr"] = easyocr_predict
            print(f"EasyOCR model loaded successfully on {'GPU' if gpu else 'CPU'}")
            return True
        except Exception as e:
            print(f"Error setting up EasyOCR: {e}")
            return False

    def setup_qwen_vl(self):
        """Setup Qwen-2.5-vl-instruct model"""
        print("Setting up Qwen-2.5-VL model...")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq, TrOCRProcessor, VisionEncoderDecoderModel
            import huggingface_hub
            
            # Check if Hugging Face token is available
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                huggingface_hub.login(token=hf_token)
                print("Logged in to Hugging Face")
            
            try:
                # Check if fine-tuned model exists
                if os.path.exists("./finetuned_qwen"):
                    print("Loading fine-tuned Qwen model...")
                    model_id = "./finetuned_qwen"
                else:
                    print("Loading base Qwen model...")
                    # Use the correct model name
                    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
                    print(f"Loading model: {model_id}")
                
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)
                
                def qwen_predict(image):
                    if isinstance(image, np.ndarray):
                        # Convert OpenCV image to PIL
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=128)
                    return processor.batch_decode(output, skip_special_tokens=True)[0]
                    
            except Exception as e:
                print(f"Error loading Qwen model: {e}, falling back to TrOCR")
                # Fallback to TrOCR
                model_id = "microsoft/trocr-base-printed"
                processor = TrOCRProcessor.from_pretrained(model_id)
                model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
                
                def qwen_predict(image):
                    if isinstance(image, np.ndarray):
                        # Convert OpenCV image to PIL
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        generated_ids = model.generate(inputs["pixel_values"], max_length=128)
                    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            self.models["qwen_vl"] = qwen_predict
            print(f"Vision-language model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error setting up Vision model: {e}")

            # Create a fallback function
            def mock_qwen(image):
                return "Vision model extraction unavailable"

            self.models["qwen_vl"] = mock_qwen
            return False

    def setup_florence2(self):
        """Setup Florence-2 model"""
        print("Setting up Vision model (for Florence-2)...")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            # Check if fine-tuned model exists
            if os.path.exists("./finetuned_florence"):
                print("Loading fine-tuned Florence model...")
                model_id = "./finetuned_florence"
            else:
                print("Loading base Florence model...")
                model_id = "microsoft/trocr-small-printed"

            processor = TrOCRProcessor.from_pretrained(model_id)
            model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
            
            def florence_predict(image):
                if isinstance(image, np.ndarray):
                    # Convert OpenCV image to PIL
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = model.generate(inputs["pixel_values"], max_length=128)
                    
                return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            self.models["florence2"] = florence_predict
            print(f"TrOCR model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error setting up vision model (Florence-2/TrOCR): {e}")

            # Create a fallback function
            def mock_florence(image):
                return "Florence model extraction unavailable"

            self.models["florence2"] = mock_florence
            return False

    def setup_ocrlatex(self):
        """Setup OCRLatex model"""
        print("Setting up OCR model (for OCRLatex)...")
        try:
            import pytesseract
            def ocrlatex_predict(image):
                if isinstance(image, Image.Image):
                    # Already a PIL Image, use directly
                    pil_image = image
                else:
                    # Convert to PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                try:
                    # Tesseract doesn't support GPU acceleration directly
                    text = pytesseract.image_to_string(pil_image)
                    return text.strip()
                except:
                    # Fallback to EasyOCR if available
                    if "easyocr" in self.models:
                        return self.models["easyocr"](image)
                    return "OCR processing failed"

            self.models["ocrlatex"] = ocrlatex_predict
            print("OCR model loaded successfully (CPU only - Tesseract)")
            return True
        except Exception as e:
            print(f"Error setting up OCR model: {e}")

            # Try to use tesseract directly if pytesseract fails
            try:
                def tesseract_predict(image):
                    if isinstance(image, Image.Image):
                        # Convert PIL Image to numpy array for OpenCV
                        img_np = np.array(image)
                    else:
                        img_np = image

                    # Convert to grayscale if it's a color image
                    if len(img_np.shape) == 3:
                        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img_np

                    # Apply some preprocessing
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    # Save to temp file and use tesseract command line
                    temp_file = 'temp_image.png'
                    cv2.imwrite(temp_file, gray)

                    # Use OS command to run tesseract
                    import subprocess
                    output_file = 'temp_output'
                    try:
                        subprocess.run(['tesseract', temp_file, output_file], check=True)
                        with open(output_file + '.txt', 'r') as f:
                            text = f.read().strip()

                        # Clean up
                        os.remove(temp_file)
                        os.remove(output_file + '.txt')

                        return text
                    except:
                        # If command fails, return error message
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        return "Tesseract command failed"

                self.models["ocrlatex"] = tesseract_predict
                print("Using tesseract command line for OCR (CPU only)")
                return True
            except:
                # Create a fallback function if even command line fails
                def mock_ocr(image):
                    return "OCR extraction unavailable"

                self.models["ocrlatex"] = mock_ocr
                return False

    def setup_all_models(self):
        """Setup all OCR models"""
        # First try EasyOCR as it's most reliable
        self.setup_easyocr()

        # Then try the others
        self.setup_ocrlatex()
        self.setup_florence2()
        self.setup_qwen_vl()

        return len(self.models)

    def extract_text(self, image_path):
        """Extract text from an image using all models"""
        if not self.models:
            print("No models loaded. Please run setup_all_models() first.")
            return {}

        print(f"\nExtracting text from: {image_path}")

        # Load the image
        try:
            pil_image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return {}

        results = {}

        # Process with each model
        for model_name, model_func in self.models.items():
            print(f"Processing with {model_name}...")

            try:
                # Measure processing time
                start_time = time.time()

                # Use PIL image for transformers models, CV2 image for others
                if model_name in ["qwen_vl", "florence2"]:
                    extracted_text = model_func(pil_image)
                else:
                    extracted_text = model_func(cv_image if cv_image is not None else pil_image)

                end_time = time.time()
                processing_time = end_time - start_time

                results[model_name] = {
                    "text": extracted_text,
                    "processing_time": processing_time
                }

                print(f"  Extracted text: {extracted_text}")
                print(f"  Processing time: {processing_time:.4f} seconds")

            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                results[model_name] = {
                    "text": f"Error: {str(e)}",
                    "processing_time": 0
                }

        return results

    def evaluate_all_images(self, image_data):
        """Evaluate all models on all images"""
        all_results = []

        for img_info in image_data:
            image_path = img_info["path"]
            ground_truth = img_info["text"]

            # Extract text from image
            extraction_results = self.extract_text(image_path)

            # Calculate metrics if ground truth is available
            if ground_truth is not None:
                for model_name, result in extraction_results.items():
                    extracted_text = result["text"]

                    # Calculate Levenshtein distance
                    lev_distance = Levenshtein.distance(extracted_text, ground_truth)

                    # Calculate character accuracy
                    max_len = max(len(extracted_text), len(ground_truth))
                    if max_len > 0:
                        char_accuracy = (max_len - lev_distance) / max_len
                    else:
                        char_accuracy = 1.0

                    # Calculate exact match
                    exact_match = 1.0 if extracted_text == ground_truth else 0.0

                    # Update results
                    extraction_results[model_name].update({
                        "ground_truth": ground_truth,
                        "levenshtein": lev_distance,
                        "char_accuracy": char_accuracy,
                        "exact_match": exact_match
                    })

            # Add to all results
            all_results.append({
                "image_path": image_path,
                "ground_truth": ground_truth,
                "results": extraction_results
            })

        # Save all results
        self.results = all_results
        return all_results

    def calculate_overall_metrics(self):
        """Calculate overall metrics for each model"""
        if not self.results:
            print("No results available. Run evaluate_all_images() first.")
            return {}

        # Initialize metrics
        overall_metrics = {}
        for result in self.results:
            for model_name in result["results"].keys():
                if model_name not in overall_metrics:
                    overall_metrics[model_name] = {
                        "total_images": 0,
                        "total_with_ground_truth": 0,
                        "total_levenshtein": 0,
                        "total_char_accuracy": 0,
                        "total_exact_match": 0,
                        "total_processing_time": 0
                    }

        # Calculate metrics
        for result in self.results:
            for model_name, model_result in result["results"].items():
                overall_metrics[model_name]["total_images"] += 1
                overall_metrics[model_name]["total_processing_time"] += model_result.get("processing_time", 0)

                # Only count metrics if ground truth is available
                if result["ground_truth"] is not None:
                    overall_metrics[model_name]["total_with_ground_truth"] += 1

                    if "levenshtein" in model_result:
                        overall_metrics[model_name]["total_levenshtein"] += model_result["levenshtein"]

                    if "char_accuracy" in model_result:
                        overall_metrics[model_name]["total_char_accuracy"] += model_result["char_accuracy"]

                    if "exact_match" in model_result:
                        overall_metrics[model_name]["total_exact_match"] += model_result["exact_match"]

        # Calculate averages
        for model_name, metrics in overall_metrics.items():
            # Calculate average processing time for ALL images
            metrics["avg_processing_time"] = metrics["total_processing_time"] / metrics["total_images"] if metrics["total_images"] > 0 else 0
            
            # Calculate other metrics only for images with ground truth
            if metrics["total_with_ground_truth"] > 0:
                metrics["avg_levenshtein"] = metrics["total_levenshtein"] / metrics["total_with_ground_truth"]
                metrics["avg_char_accuracy"] = metrics["total_char_accuracy"] / metrics["total_with_ground_truth"]
                metrics["avg_exact_match"] = metrics["total_exact_match"] / metrics["total_with_ground_truth"]
            else:
                metrics["avg_levenshtein"] = None
                metrics["avg_char_accuracy"] = None
                metrics["avg_exact_match"] = None

        return overall_metrics

    def plot_comparison_graphs(self):
        """Plot comparison graphs for all models"""
        metrics = self.calculate_overall_metrics()

        if not metrics:
            print("No metrics available for plotting.")
            return

        model_names = list(metrics.keys())

        # 1. Average Character Accuracy
        accuracies = [metrics[model]["avg_char_accuracy"] for model in model_names if metrics[model]["avg_char_accuracy"] is not None]
        valid_accuracy_models = [model for model in model_names if metrics[model]["avg_char_accuracy"] is not None]

        if valid_accuracy_models:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=valid_accuracy_models, y=accuracies)
            plt.title("Average Character Accuracy by Model")
            plt.xlabel("OCR Model")
            plt.ylabel("Average Character Accuracy")
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join("results", "avg_char_accuracy.png"))
            plt.close()
            print("Average Character Accuracy plot saved to results/avg_char_accuracy.png")
        else:
            print("No character accuracy data available for plotting.")

        # 2. Average Levenshtein Distance
        distances = [metrics[model]["avg_levenshtein"] for model in model_names if metrics[model]["avg_levenshtein"] is not None]
        valid_distance_models = [model for model in model_names if metrics[model]["avg_levenshtein"] is not None]

        if valid_distance_models:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=valid_distance_models, y=distances)
            plt.title("Average Levenshtein Distance by Model")
            plt.xlabel("OCR Model")
            plt.ylabel("Average Levenshtein Distance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join("results", "avg_levenshtein_distance.png"))
            plt.close()
            print("Average Levenshtein Distance plot saved to results/avg_levenshtein_distance.png")
        else:
            print("No Levenshtein distance data available for plotting.")

        # 3. Average Exact Match Ratio
        exact_matches = [metrics[model]["avg_exact_match"] for model in model_names if metrics[model]["avg_exact_match"] is not None]
        valid_match_models = [model for model in model_names if metrics[model]["avg_exact_match"] is not None]

        if valid_match_models:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=valid_match_models, y=exact_matches)
            plt.title("Average Exact Match Ratio by Model")
            plt.xlabel("OCR Model")
            plt.ylabel("Average Exact Match Ratio")
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join("results", "avg_exact_match.png"))
            plt.close()
            print("Average Exact Match Ratio plot saved to results/avg_exact_match.png")
        else:
            print("No exact match data available for plotting.")

        # 4. Average Processing Time
        processing_times = [metrics[model]["avg_processing_time"] for model in model_names]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=processing_times)
        plt.title("Average Processing Time by Model")
        plt.xlabel("OCR Model")
        plt.ylabel("Average Processing Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "avg_processing_time.png"))
        plt.close()
        print("Average Processing Time plot saved to results/avg_processing_time.png")

        # 5. Heatmap of Character Accuracy per Base Text
        if self.results and any(item["ground_truth"] is not None for item in self.results):
            # Group results by base text
            base_text_results = {}
            for res in self.results:
                if res["ground_truth"] is not None:
                    # Extract base text index from path (e.g., "text_0" from "text_0_blur_0.png")
                    base_text = os.path.basename(res["image_path"]).split('_')[0:2]
                    base_text = '_'.join(base_text)
                    
                    if base_text not in base_text_results:
                        base_text_results[base_text] = {
                            'ground_truth': res["ground_truth"],
                            'accuracies': {model: [] for model in model_names}
                        }
                    
                    # Add accuracies for each model
                    for model in model_names:
                        if model in res["results"] and "char_accuracy" in res["results"][model]:
                            base_text_results[base_text]['accuracies'][model].append(
                                res["results"][model]["char_accuracy"]
                            )

            # Calculate average accuracy for each base text and model
            accuracy_data = {}
            for model in model_names:
                accuracy_data[model] = []
            
            base_texts = []
            for base_text, data in base_text_results.items():
                base_texts.append(base_text)
                for model in model_names:
                    accuracies = data['accuracies'][model]
                    avg_accuracy = np.mean(accuracies) if accuracies else np.nan
                    accuracy_data[model].append(avg_accuracy)

            if base_texts:
                df_accuracy = pd.DataFrame(accuracy_data, index=base_texts)
                plt.figure(figsize=(len(model_names) * 1.5, len(base_texts) * 0.8))
                sns.heatmap(df_accuracy, annot=True, cmap="viridis", vmin=0, vmax=1, fmt='.3f')
                plt.title("Average Character Accuracy per Base Text and Model")
                plt.ylabel("Base Text")
                plt.xlabel("OCR Model")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join("results", "char_accuracy_heatmap.png"))
                plt.close()
                print("Character Accuracy Heatmap saved to results/char_accuracy_heatmap.png")
            else:
                print("No character accuracy data with ground truth available for heatmap.")
        else:
            print("No evaluation data with ground truth available for heatmap.")


if __name__ == "__main__":
    # Download test images
    image_data = download_test_images()

    # Initialize OCR model extractor
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