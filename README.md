# NeuroScan AI — Brain Tumor Classifier

NeuroScan AI is an Explainable AI (XAI) deep learning system designed to classify brain tumors from MRI scans. Built with **EfficientNet-B0/B4**, it classifies MRI scans into 4 perfectly balanced categories: **Glioma, Meningioma, No Tumor, and Pituitary**. It features a modern FastAPI-powered dashboard that uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate visual heatmaps explaining *why* the model made its prediction.

## Key Features
- **Highly Accurate**: Based on EfficientNet architecture, heavily utilizing Transfer Learning (ImageNet weights).
- **Explainable AI (XAI)**: Includes Grad-CAM overlays to visually localize tumor regions that drove the model's decision.
- **FastAPI Backend**: REST API handling inference, Grad-CAM generation, and batch predictions.
- **Dynamic Dashboard**: A beautiful, responsive frontend with a dark glassmorphism design to display predictions, confidence scores, and Grad-CAM overlays.
- **CLI Inference**: Evaluate and visualize single scans or batch directories directly from the terminal.

## Dataset
The model is trained on a perfectly balanced subset of the Brain MRI Tumor dataset (7,200 images in total):
- 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- **Training Set**: 5,600 images (1,400 per class)
- **Testing Set**: 1,600 images (400 per class)

## Project Structure
- `config.py`: Centralized configuration for hyperparameters, paths, and class labels.
- `model.py`: EfficientNet-B0/B4 classifier architecture.
- `dataset.py`: PyTorch dataset definitions and augmentation pipelines.
- `train.py`: Training script with a two-phase fine-tuning approach.
- `inference.py`: CLI tool for single/batch predictions and Grad-CAM visualizations.
- `evaluate.py`: Generates confusion matrix, ROC curves, and evaluation metrics.
- `gradcam_utils.py`: Utilities for generating Grad-CAM heatmaps.
- `dashboard/`: FastAPI application and frontend static files.

For an in-depth technical dive into the architecture, model layers, and Grad-CAM logic, please see [DOCUMENTATION.md](./DOCUMENTATION.md).

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed. Navigate to the project directory and run:

```powershell
# Install core packages for CPU environment (if you have CUDA, adjust the torch index-url accordingly)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install timm grad-cam Pillow opencv-python numpy matplotlib seaborn scikit-learn fastapi "uvicorn[standard]" python-multipart
```

Alternatively, you can just install from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 2. Training the Model (Optional)
If you want to train the model from scratch:
```powershell
# Pre-download EfficientNet weights to avoid HuggingFace download hangs
python -c "import timm; timm.create_model('efficientnet_b0', pretrained=True); print('Done')"

# Start training (auto-detects CUDA/CPU)
# -u flag is recommended on Windows to ensure live output
python -u train.py --batch-size 16 --epochs 50
```
*Note: A pre-trained checkpoint should be generated in `outputs/checkpoints/best_model.pth` before running inference or the dashboard.*

### 3. Running Evaluation
To test the model on the held-out Testing set and generate metrics (Confusion Matrix, ROC curves):
```powershell
python -u evaluate.py
```
Check the `outputs/eval/` directory for the results.

### 4. CLI Inference
You can use the CLI script to classify images and optionally generate Grad-CAM visualizations:

```powershell
# Single image prediction
python inference.py --source "Brain MRI Tumor/Testing/glioma/Te-gl_0010.jpg"

# Predict with a Grad-CAM overlay heatmap
python inference.py --source "Brain MRI Tumor/Testing/glioma/Te-gl_0010.jpg" --visualise

# Generate an All-Class 5-panel Grad-CAM visualization
python inference.py --source "Brain MRI Tumor/Testing/glioma/Te-gl_0010.jpg" --all-classes

# Batch inference (outputs to a CSV)
python inference.py --source "Brain MRI Tumor/Testing/" --output results.csv
```
Visualizations are saved to the `outputs/visualizations/` folder.

### 5. Launch the Web Dashboard
A beautiful web interface is available for interactive diagnosis.

```powershell
# Navigate to the dashboard directory
cd dashboard

# Start the FastAPI server using Uvicorn
python -m uvicorn app:app --reload --port 8000
```
Open your browser and navigate to: **http://localhost:8000**
