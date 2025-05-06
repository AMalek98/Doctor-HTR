# Medical Document Text Recognition Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMalek98/Doctor-HTR.git)

Notebook-based implementation of medical document processing pipeline with YOLOv8 localization and TrOCR handwriting recognition.

## Project Structure
### Notebook Workflow:
1. `gray_scale_check.ipynb`  
   - Initial image analysis and grayscale conversion validation
   - Histogram equalization verification

2. `roi_preprocess.ipynb`  
   - **Crucial preprocessing** for YOLO success
   - ROI extraction and image enhancement
   - Noise reduction techniques

3. `yolo_v8n.ipynb`  
   - Two-phase YOLOv8n training:
     1. Initial training on ROI-enhanced images
     2. Fine-tuning on real medical documents
   - Achieved **88% mAP** for word localization

4. `bounding_box_extraction.ipynb`  
   - YOLO prediction parsing
   - Coordinate conversion (YOLO ↔ Pascal VOC)
   - `text_result.txt` generation

5. `labeling_all.ipynb`  
   - Dataset preparation for TrOCR training
   - Image-text pairing
   - Train/validation split creation

6. `microsoft_trocr_base1.ipynb`  
   - TrOCR-base model fine-tuning
   - Medicine name recognition specialization
   - Achieved **0.29 CER**

## Key Achievements

✅ **88% mAP** for word localization using YOLOv8n  
✅ **0.29 CER** on medicine names with customized TrOCR  
✅ **50% faster annotation** through preprocessing optimization  
✅ Robust two-phase training strategy for detection model


## Installation

```bash
git clone https://github.com/AMalek98/Doctor-HTR.git
pip install -r requirements.txt
