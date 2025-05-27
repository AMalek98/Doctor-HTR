# DOCTOR_HTR: Medical Handwriting Text Recognition

## Project Description

This repository contains a research project focused on developing and evaluating handwritten text recognition capabilities for medical documents. This work represents a proof of concept study investigating the feasibility of automated handwriting recognition in healthcare documentation using modern computer vision and OCR techniques.

The project demonstrates a complete pipeline from object detection to text recognition, specifically tailored for the challenges inherent in medical handwriting analysis.

## Research Objectives

- **Evaluate YOLO's performance** for detecting text regions in medical handwriting samples
- **Develop a preprocessing pipeline** optimized for medical document images with ROI extraction
- **Implement TrOCR fine-tuning** for medical terminology recognition and transcription
- **Create a foundation** for future development of comprehensive medical HTR systems
- **Demonstrate proof of concept** for automated medical document digitization workflows

## Key Findings

- **YOLO demonstrates strong generalization capabilities** on unseen medical handwriting samples, achieving 88% mAP for word localization
- **The developed ROI preprocessing pipeline** significantly improves detection accuracy by focusing on text regions
- **TrOCR fine-tuning** achieved 0.29 CER (Character Error Rate) on medical terminology recognition
- **Results indicate promising potential** for real-world medical document processing applications
- **The model shows robust performance** on documents it wasn't trained on, as evidenced in `/data_samples/result_test2.jpg`

## Project Status

**This is a research project and proof of concept.** The code and models provided serve as a foundation for future development and should be considered experimental. The implementation demonstrates feasibility rather than production-ready deployment.

## Repository Structure

### Jupyter Notebooks (Processing Pipeline)

#### 1. Data Analysis and Preprocessing
- **`gray_scale_check.ipynb`** - Grayscale level analysis for background consistency determination
- **`roi_preprocess.ipynb`** - Crucial ROI extraction and image enhancement pipeline  
- **`word_freq.ipynb`** - Word frequency analysis and dataset statistics

#### 2. Object Detection (YOLO Training)
- **`Yolo_v8n.ipynb`** - Complete YOLOv8n training pipeline with two-phase approach
  - Initial training on ROI-enhanced images  
  - Fine-tuning on real medical documents
  - Achieved **88% mAP** for word localization

#### 3. Dataset Preparation
- **`bounding_box_extraction.ipynb`** - YOLO prediction parsing and coordinate conversion
- **`combining_bbox.ipynb`** - Dataset consolidation for unified structure
- **`labeling_all.ipynb`** - TrOCR training dataset preparation and train/validation splitting

#### 4. Text Recognition (OCR)
- **`microsoft_trocr_base1.ipynb`** - TrOCR-base model fine-tuning for medical terminology
  - Specialized medicine name recognition
  - Achieved **0.29 CER** on medical text

### Data Samples
- **`data_samples/`** - Representative images showcasing model performance
  - `result_test2.jpg` - Demonstrates model generalization on unseen documents
  - Various line samples showing detection capabilities

### Configuration
- **`requirements.txt`** - Python dependencies for the research environment
- **`doctor_htr_guide.md`** - Comprehensive refactoring and development guide

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Sufficient storage for medical document datasets

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/AMalek98/Doctor-HTR.git
cd Doctor-HTR

# Install dependencies
pip install -r requirements.txt

# Additional setup for Google Colab (if using)
# Mount Google Drive for dataset access
```

### Required Dependencies
- **PyTorch** >= 2.0.0 - Deep learning framework
- **Ultralytics** >= 8.0.0 - YOLO implementation
- **Transformers** >= 4.30.0 - TrOCR model access
- **OpenCV** >= 4.7.0 - Image processing
- **Matplotlib** >= 3.7.0 - Visualization
- **Pandas** >= 2.0.0 - Data manipulation

## Usage Instructions

### Step 1: Data Preparation
1. **Grayscale Analysis**: Run `gray_scale_check.ipynb` to determine optimal background parameters
2. **ROI Preprocessing**: Execute `roi_preprocess.ipynb` for image enhancement
3. **Update paths**: Replace all "enter your path here" placeholders with your dataset paths

### Step 2: Object Detection Training
1. **YOLO Training**: Execute `Yolo_v8n.ipynb` for text detection model training
   - Ensure dataset configuration YAML is properly set up
   - Monitor training progress and validation metrics
   - Save best performing model weights

### Step 3: Dataset Preparation for OCR
1. **Bounding Box Extraction**: Run `bounding_box_extraction.ipynb` to extract text regions
2. **Dataset Consolidation**: Execute `combining_bbox.ipynb` to create unified dataset
3. **Label Preparation**: Use `labeling_all.ipynb` for TrOCR training data setup

### Step 4: Text Recognition Training
1. **TrOCR Fine-tuning**: Execute `microsoft_trocr_base1.ipynb` for text recognition
   - Configure training parameters for your dataset size
   - Monitor character error rate (CER) improvements
   - Save fine-tuned model for inference

### Step 5: Analysis and Evaluation
1. **Performance Assessment**: Evaluate both detection and recognition performance
2. **Word Frequency Analysis**: Run `word_freq.ipynb` for dataset insights
3. **Results Visualization**: Generate sample outputs for documentation

## Technical Architecture

### Two-Phase Training Strategy
1. **Phase 1**: Initial YOLO training on ROI-preprocessed images
2. **Phase 2**: Fine-tuning on real medical documents for domain adaptation

### ROI Preprocessing Benefits
- **Background noise reduction** for improved detection focus
- **Consistent grayscale levels** (212.85 intensity) for training stability
- **Enhanced text visibility** through contrast optimization
- **Reduced false positives** by eliminating irrelevant image regions

### Model Performance Metrics
- **Object Detection**: 88% mAP@0.5 for word localization
- **Text Recognition**: 0.29 CER on medical terminology
- **Processing Speed**: 50% faster annotation through preprocessing optimization
- **Generalization**: Robust performance on unseen document types

## Research Impact and Applications

### Healthcare Documentation
- **Clinical Workflow Integration**: Potential for automated transcription systems
- **Electronic Health Records**: Digitization of handwritten medical notes
- **Prescription Processing**: Automated medicine name recognition and verification
- **Archive Digitization**: Conversion of historical medical documents

### Technical Contributions
- **Domain-Specific Preprocessing**: ROI extraction methodology for medical documents
- **Transfer Learning Approach**: YOLO adaptation for medical handwriting detection
- **Multi-Modal Pipeline**: Integration of object detection and text recognition
- **Evaluation Framework**: Metrics and benchmarks for medical HTR assessment

## Future Work

### Immediate Enhancements
- **Multi-language Support**: Extension to diverse medical terminology languages
- **Real-time Processing**: Optimization for clinical workflow integration
- **Mobile Deployment**: Lightweight models for point-of-care applications
- **Quality Assessment**: Automated confidence scoring for recognition results

### Research Directions
- **Synthetic Data Generation**: Augmentation techniques for training data expansion
- **Few-shot Learning**: Adaptation to new medical specialties with limited data
- **Temporal Modeling**: Integration of contextual information for improved accuracy
- **Privacy-Preserving ML**: Federated learning approaches for sensitive medical data

### Production Considerations
- **Scalability Testing**: Performance evaluation on large document collections
- **Integration APIs**: RESTful services for healthcare system integration
- **Compliance Framework**: HIPAA and healthcare regulation adherence
- **Error Handling**: Robust failure modes and recovery mechanisms

## Limitations and Considerations

### Current Scope
- **Proof of Concept**: Not intended for production medical use without further validation
- **Dataset Size**: Limited to available training samples - performance may vary on different handwriting styles
- **Language Coverage**: Primarily focused on English medical terminology
- **Computational Requirements**: GPU-intensive training pipeline

### Recommended Validation
- **Clinical Testing**: Validation with healthcare professionals before deployment
- **Accuracy Assessment**: Comprehensive evaluation on diverse medical document types
- **Error Analysis**: Systematic review of recognition failures and improvement opportunities
- **Regulatory Review**: Compliance assessment for medical device regulations

## Contributing and Collaboration

This research project welcomes collaboration from:
- **Healthcare Professionals**: Domain expertise and validation requirements
- **Computer Vision Researchers**: Technical improvements and methodology enhancements  
- **Medical Informaticists**: Integration strategies and workflow optimization
- **Healthcare Technology Companies**: Production deployment and scaling insights

## Citation and Acknowledgments

If you use this work in your research, please cite:

```
DOCTOR_HTR: Medical Handwriting Text Recognition Research Project
Proof of Concept for Automated Medical Document Processing
[Repository URL]
```

### Acknowledgments
- **Ultralytics**: YOLOv8 implementation and documentation
- **Microsoft Research**: TrOCR model and transformer architecture
- **Medical Community**: Inspiration and domain requirements for healthcare digitization

---

*This implementation represents a significant step toward automated medical document processing, demonstrating strong potential for improving healthcare documentation workflows through advanced computer vision and NLP techniques.*