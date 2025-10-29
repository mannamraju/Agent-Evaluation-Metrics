# Agent Evaluation Metrics — Evaluation Metrics for Healthcare Agentic Applications and Chest X‑Ray Image Analysis using Open Source Models

This repository provides comprehensive evaluation metrics for healthcare AI applications with a focus on:

* **Healthcare Agent Evaluation**: Specialized metrics for evaluating AI agents in medical contexts
* **Medical Image Analysis**: Open source models for chest X-ray analysis and interpretation
* **Multi-modal Medical AI**: Vision-language models for medical image understanding
* **Lightweight Model Deployment**: Optimized models for consumer hardware (RTX 4050/6GB VRAM)

## Core Evaluation Metrics

### Text-based Metrics

* **BLEU-4** - N-gram overlap for report generation quality
  * **Input Data**: Generated medical reports (plain text)
  * **Ground Truth**: Reference radiology reports from expert radiologists
  * **Data Format**: CSV files with "report" column containing text reports
  * **Use Case**: Measuring linguistic similarity and fluency of generated reports

* **BERTScore** - Semantic similarity using contextualized embeddings  
  * **Input Data**: Generated medical reports (plain text)
  * **Ground Truth**: Reference radiology reports with clinical terminology
  * **Data Format**: CSV files with "report" column, uses Bio_ClinicalBERT embeddings
  * **Use Case**: Capturing semantic meaning beyond surface-level text similarity

* **ROUGE-L** - Longest common subsequence overlap
  * **Input Data**: Generated medical reports (plain text)
  * **Ground Truth**: Reference reports focusing on content recall
  * **Data Format**: CSV files with "report" column
  * **Use Case**: Evaluating content coverage and recall of important medical findings

* **Composite RadCliQ** - Specialized radiology report quality metric
  * **Input Data**: Generated radiology reports with structured findings
  * **Ground Truth**: Expert-annotated radiology reports with quality scores
  * **Data Format**: JSON format with structured report sections
  * **Use Case**: Comprehensive radiology-specific quality assessment

### Medical AI Metrics

* **Semantic Embedding Analysis** - Clinical concept similarity evaluation
  * **Input Data**: Generated reports with medical terminology
  * **Ground Truth**: Reference reports with validated clinical concepts
  * **Data Format**: Text files processed through medical NLP pipelines
  * **Use Case**: Measuring clinical concept alignment and medical accuracy

* **RadGraph Evaluation** - Medical entity and relation extraction assessment
  * **Input Data**: Generated radiology reports (plain text)
  * **Ground Truth**: RadGraph-annotated reports with entity-relation graphs
  * **Data Format**: Text files with corresponding RadGraph JSON annotations
  * **Use Case**: Evaluating clinical reasoning and medical entity relationships

* **Perplexity Metrics** - Language model uncertainty quantification
  * **Input Data**: Generated medical text sequences
  * **Ground Truth**: Medical corpus for language model training/validation
  * **Data Format**: Plain text files with medical terminology
  * **Use Case**: Measuring fluency and medical language modeling quality

* **Bounding Box IoU** - Anatomical localization accuracy
  * **Input Data**: Generated bounding boxes on chest X-ray images
  * **Ground Truth**: Expert-annotated bounding boxes for anatomical regions
  * **Data Format**: JSON files with bbox coordinates + corresponding DICOM/PNG images
  * **Use Case**: Measuring spatial accuracy of anatomical region detection

### Agent-specific Metrics

* **Multi-turn Conversation Evaluation** - Healthcare dialogue assessment
  * **Input Data**: Generated multi-turn conversations between AI agent and users
  * **Ground Truth**: Expert-validated medical consultation dialogues
  * **Data Format**: JSON files with conversation history and turn-level annotations
  * **Use Case**: Evaluating conversational coherence in medical contexts

* **Clinical Decision Support Metrics** - Decision accuracy and reasoning evaluation
  * **Input Data**: AI agent diagnostic suggestions and reasoning chains
  * **Ground Truth**: Expert clinical decisions with documented reasoning
  * **Data Format**: Structured JSON with decision trees and evidence linking
  * **Use Case**: Measuring diagnostic accuracy and clinical reasoning quality

* **Multimodal Alignment** - Image-text correspondence in medical contexts
  * **Input Data**: Generated image descriptions paired with medical images
  * **Ground Truth**: Expert-verified image-report pairs with alignment scores
  * **Data Format**: Image files (DICOM/PNG) + corresponding text reports + alignment labels
  * **Use Case**: Ensuring visual findings match textual descriptions accurately


## Table of Contents

* [Prerequisites](#prerequisites)
* [Open Source Medical Models](#open-source-medical-models)
* [System Requirements](#system-requirements)
* [Notebooks and Demos](#notebooks-and-demos)
* [Usage](#usage)
* [Evaluation Framework](#evaluation-framework)
* [License](#license)
* [Citing](#citing)


## Prerequisites

To install the dependencies, run the following command with Python 3.8+:

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended):
```bash
pip install -r requirements-azure.txt  # Includes CUDA dependencies
```

## Open Source Medical Models

Our repository integrates lightweight, state-of-the-art open source medical AI models optimized for consumer hardware:

### Vision-Language Models

* **BiomedCLIP** - Medical image-text understanding model
  - Memory: ~2-3GB VRAM
  - Capabilities: Zero-shot pathology detection, medical image classification
  - Source: Microsoft/BiomedCLIP

* **OpenAI CLIP (Medical Fine-tuned)** - General purpose vision-language model adapted for medical use
  - Memory: ~1-2GB VRAM  
  - Capabilities: Medical image analysis, report generation assistance

### Pathology Detection Models

* **CheXNet (DenseNet-121)** - Stanford's chest X-ray pathology classifier
  - Memory: ~1.5GB VRAM
  - Detects: 14 different chest pathologies including pneumonia, pneumothorax, edema
  - Accuracy: High (AUC > 0.8 for most pathologies)

* **NIH ChestX-ray14** - Multi-label chest pathology classification
  - Memory: ~2GB VRAM
  - Pathologies: 14 different conditions
  - Performance: Strong baseline for chest X-ray analysis

### Segmentation Models

* **MedSAM** - Medical Segment Anything Model
  - Memory: ~4-5GB VRAM (fits on RTX 4050)
  - Purpose: General medical image segmentation
  - Segments: Any anatomical structure with prompts

* **SAM-Med2D** - 2D Medical segmentation specialist
  - Memory: ~3-4GB VRAM
  - Purpose: Organs, lesions, anatomical regions
  - Optimized: Faster inference than general SAM

### Lightweight Efficient Models

* **MobileNet Medical Classifiers**
  - Memory: <1GB VRAM
  - Speed: Very fast (real-time capable)
  - Use case: Resource-constrained deployments

* **EfficientNet Medical Models**
  - Memory: ~1-2GB VRAM
  - Balance: High accuracy with efficiency
  - Training: Pre-trained on RadImageNet

## System Requirements

### Minimum Requirements
- **GPU**: 2GB VRAM (basic models)
- **RAM**: 8GB system memory
- **Python**: 3.8 or higher
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+

### Recommended Configuration (Tested)
- **GPU**: NVIDIA RTX 4050 (6GB VRAM) or equivalent
- **RAM**: 16GB system memory
- **Storage**: 10GB free space for model weights
- **CUDA**: 11.8+ for GPU acceleration

### Hardware Compatibility
- ✅ **RTX 40 Series** (4050, 4060, 4070+) - All models supported
- ✅ **RTX 30 Series** (3060+) - All models supported  
- ✅ **GTX 16 Series** (1660+) - Most models supported
- ⚠️ **Integrated Graphics** - CPU-only lightweight models

## Notebooks and Demos

### Interactive Jupyter Notebooks

#### `oss_models_evals/medical_imaging_demo.ipynb`
**Comprehensive Medical AI Model Showcase**
- **Purpose**: Interactive demonstration of all supported medical AI models
- **Features**: 
  - Real-time GPU memory monitoring
  - Model performance comparisons
  - Actual chest X-ray image analysis
  - Visual results with confidence scores
- **Models Demonstrated**: BiomedCLIP, CheXNet variants, segmentation models
- **Runtime**: ~10-15 minutes for full notebook execution
- **Data**: Includes sample chest X-ray images with pathologies

### Python Demo Scripts

#### `oss_models_evals/medical_imaging_demo.py`
**Command-line Medical Model Demo**
- **Purpose**: Quick model testing and validation
- **Usage**: `python oss_models_evals/medical_imaging_demo.py`
- **Features**: GPU compatibility check, model recommendations

#### `medical_model_alternatives.py`
**Model Selection Guide**
- **Purpose**: Hardware-specific model recommendations
- **Output**: Compatibility matrix for your system
- **Usage**: `python medical_model_alternatives.py`

### Evaluation Demos

#### `CXRMetric/metrics/bleu/run_bleu_demo.py`
**BLEU Metric Evaluation**
- **Purpose**: Text similarity evaluation for generated reports
- **Features**: Multiple BLEU variants, detailed scoring breakdown
- **Output**: Timestamped evaluation reports

#### `CXRMetric/metrics/bertscore/run_bertscore_demo.py`
**BERTScore Semantic Evaluation**  
- **Purpose**: Semantic similarity using transformer embeddings
- **Features**: Contextual understanding, clinical relevance scoring
- **Models**: Bio_ClinicalBERT, BioBERT variants

## Usage

### Quick Start

#### Running Evaluation Metrics

```python
from CXRMetric.run_eval import calc_metric

# Evaluate chest X-ray reports
calc_metric(
    gt_reports="reports/gt_reports.csv", 
    predicted_reports="reports/predicted_reports.csv",
    out_file="outputs/metrics/evaluation_results.json",
    use_idf=True
)
```

#### Medical Image Analysis

```python
# Run medical imaging demo on actual chest X-rays
python oss_models_evals/medical_imaging_demo.py

# Interactive notebook analysis
jupyter notebook oss_models_evals/medical_imaging_demo.ipynb
```

### Data Format Requirements

#### Standard CSV Format for Text Metrics

Ground Truth and Predicted reports must be arranged in the same order in a column named "report" in two CSV files. The CSVs should also contain a corresponding "study_id" column that contains unique identifiers for the reports.

**Required CSV Structure:**

```csv
study_id,report
CXR001,"Normal chest radiograph. No acute cardiopulmonary process."
CXR002,"Bilateral lower lobe infiltrates consistent with pneumonia."
CXR003,"Cardiomegaly with pulmonary vascular congestion."
```

#### Specialized Data Formats by Metric Type

**Text-based Metrics (BLEU, BERTScore, ROUGE):**
* **File Format**: CSV files with UTF-8 encoding
* **Required Columns**: `study_id`, `report`
* **Optional Columns**: `patient_id`, `study_date`, `modality`
* **Text Requirements**: Clean text without special formatting, medical terminology preserved
* **Example Location**: `reports/gt_reports.csv`, `reports/predicted_reports.csv`

**Medical Entity Metrics (RadGraph):**
* **File Format**: JSON files with text and annotation pairs
* **Structure**: 
  ```json
  {
    "study_id": "CXR001",
    "text": "Chest X-ray shows bilateral pneumonia...",
    "entities": [{"text": "pneumonia", "label": "FINDING", "start": 25, "end": 34}],
    "relations": [{"head": 0, "tail": 1, "relation": "LOCATED_AT"}]
  }
  ```

**Bounding Box Metrics (IoU):**
* **Image Format**: DICOM (.dcm) or PNG/JPEG (.png, .jpg)
* **Annotation Format**: JSON with bounding box coordinates
* **Structure**:
  ```json
  {
    "study_id": "CXR001",
    "image_path": "images/CXR001.png",
    "annotations": [
      {"bbox": [x1, y1, x2, y2], "label": "pneumonia", "confidence": 0.95}
    ]
  }
  ```

**Multimodal Data (Image + Text):**
* **Images**: DICOM format preferred, PNG/JPEG acceptable
* **Reports**: Paired CSV or JSON format
* **Structure**: 
  ```json
  {
    "study_id": "CXR001",
    "image_path": "images/CXR001.dcm",
    "frontal_view": "images/CXR001_frontal.dcm",
    "lateral_view": "images/CXR001_lateral.dcm",
    "report": "Frontal and lateral chest radiographs...",
    "findings": ["Normal heart size", "Clear lungs"]
  }
  ```

**Agent Conversation Data:**
* **Format**: JSON with conversation history
* **Structure**:
  ```json
  {
    "conversation_id": "CONV001",
    "turns": [
      {"role": "user", "content": "I have chest pain", "timestamp": "2024-01-01T10:00:00Z"},
      {"role": "agent", "content": "Can you describe the pain?", "timestamp": "2024-01-01T10:00:05Z"}
    ],
    "ground_truth": {
      "diagnosis": "Acute coronary syndrome",
      "confidence": 0.85,
      "reasoning": "Patient presents with typical chest pain..."
    }
  }
  ```

#### Data Quality Requirements

**Text Data:**
* Minimum report length: 10 words
* Maximum report length: 2000 words
* Language: English (medical terminology)
* Encoding: UTF-8
* Special characters: Medical symbols preserved (±, >, <, etc.)

**Image Data:**
* Resolution: Minimum 512x512 pixels for chest X-rays
* Bit depth: 16-bit preferred for DICOM, 8-bit acceptable for PNG
* Format requirements: DICOM with proper medical metadata
* File size: < 50MB per image for processing efficiency

**Annotation Quality:**
* Inter-annotator agreement: κ > 0.8 for bounding boxes
* Expert validation: All ground truth reviewed by board-certified radiologists
* Consistency checks: Automated validation for format compliance
* Completeness: All required fields populated, no missing critical annotations

### Configuration

In `config.py`, set:
* `GT_REPORTS` - Path to ground truth reports CSV
* `PREDICTED_REPORTS` - Path to predicted reports CSV  
* `OUT_FILE` - Desired output path for metric scores

## Evaluation Framework

### Comprehensive Medical AI Evaluation

This repository implements a multi-layered evaluation approach specifically designed for healthcare agentic applications:

#### Layer 1: Text Generation Quality
* **BLEU-4**: Measures n-gram overlap between generated and reference reports
* **ROUGE-L**: Evaluates longest common subsequence recall
* **BERTScore**: Assesses semantic similarity using contextual embeddings
* **Perplexity**: Quantifies language model uncertainty and fluency

#### Layer 2: Clinical Accuracy  
* **RadGraph F1**: Evaluates medical entity and relation extraction
* **CheXpert Label Accuracy**: Multi-label pathology classification performance
* **Semantic Embedding Alignment**: Clinical concept similarity in vector space
* **Composite RadCliQ**: Specialized radiology report quality metric

#### Layer 3: Visual-Textual Alignment
* **Bounding Box IoU**: Anatomical localization accuracy
* **Region-Report Correspondence**: Alignment between highlighted regions and text
* **Multi-modal Consistency**: Vision-language model coherence evaluation

#### Layer 4: Agent-Specific Metrics
* **Conversation Coherence**: Multi-turn dialogue evaluation in medical contexts
* **Clinical Decision Support**: Accuracy of diagnostic suggestions and reasoning
* **Safety and Bias Detection**: Identification of potential harmful outputs

### Model Architecture Support

Our evaluation framework supports various medical AI architectures:

* **Transformer-based Models**: BERT, GPT, T5 variants fine-tuned for medical text
* **Vision-Language Models**: CLIP, BLIP, LLaVA adapted for medical imaging
* **Multimodal Architectures**: Joint image-text encoders for radiology reports
* **Agent Frameworks**: LangChain, AutoGen, CrewAI for healthcare applications

### Class Hierarchy Architecture

The evaluation framework follows a consistent inheritance pattern based on the `BaseEvaluator` abstract class:

```text
BaseEvaluator (Abstract Base Class)
├── compute_metric() → Dict[str, float]
├── get_metric_columns() → List[str]
├── align_predictions_and_ground_truth()
└── validate_inputs()

Core Metric Implementations:
├── BLEUEvaluator
│   ├── Inherits: BaseEvaluator
│   ├── Metric: BLEU-1, BLEU-2, BLEU-3, BLEU-4
│   └── Data: Text reports (CSV format)
│
├── BERTScoreEvaluator  
│   ├── Inherits: BaseEvaluator
│   ├── Metric: Precision, Recall, F1 using BERT embeddings
│   └── Data: Text reports with Bio_ClinicalBERT
│
├── ROUGEEvaluator
│   ├── Inherits: BaseEvaluator
│   ├── Metric: ROUGE-1, ROUGE-2, ROUGE-L
│   └── Data: Text reports (CSV format)
│
├── CompositeMetricEvaluator
│   ├── Inherits: BaseEvaluator
│   ├── Metric: Combined multi-metric scoring
│   └── Data: Aggregated results from multiple evaluators
│
├── BoundingBoxEvaluator
│   ├── Inherits: BaseEvaluator
│   ├── Metric: IoU, Precision, Recall for bounding boxes
│   └── Data: JSON with bbox coordinates
│
└── PerplexityEvaluator
    ├── Inherits: BaseEvaluator
    ├── Metric: Language model perplexity scores
    └── Data: Text reports with language model evaluation

Optional Model-Dependent Metrics (Shim Pattern):
├── SemanticEmbeddingEvaluator
│   ├── Inherits: BaseEvaluator (with graceful degradation)
│   ├── Metric: Cosine similarity of clinical embeddings
│   ├── Data: Medical text with semantic concept extraction
│   └── Dependencies: Optional - falls back if models unavailable
│
└── RadGraphEvaluator
    ├── Inherits: BaseEvaluator (with graceful degradation)
    ├── Metric: Entity-relation graph similarity
    ├── Data: RadGraph-annotated reports (JSON format)
    └── Dependencies: Optional - requires RadGraph NLP pipeline
```

**Key Design Principles:**

* **Consistent Interface**: All evaluators implement the same abstract methods
* **Graceful Degradation**: Optional metrics use shim pattern for missing dependencies
* **Modular Design**: Easy to add new metrics by extending BaseEvaluator
* **Data Flexibility**: Supports multiple input formats (CSV, JSON, text files)

### Performance Benchmarks

The repository includes benchmark results on standard datasets:

#### MIMIC-CXR Test Set (Findings Generation)

* CheXpert F1-5 (micro): 59.7%
* ROUGE-L: 39.1%  
* BLEU-4: 23.7%

#### GR-Bench Test Set (Grounded Reporting)

* ROUGE-L: 56.6%
* Box-Completion Precision: 71.5%
* Box-Completion Recall: 82.0%

### Supported Model Types

This repository currently supports the following metrics by default:

* **Standard NLG Metrics**: BLEU, BERTScore, ROUGE  
* **Medical Imaging Metrics**: Bounding-box IoU, pathology classification
* **Lightweight Deployment**: CPU-friendly versions of all metrics
* **Optional Advanced Metrics**: CheXbert, RadGraph (requires model downloads)

**Note**: Model-based metrics that require external checkpoints (CheXbert, RadGraph) are organized in the `CXRMetric/metrics/optional/` directory and include graceful fallbacks when models are unavailable.

## Evaluation Metrics and Protocol

This repository computes a set of metrics that together assess language fidelity, clinical correctness, and image-grounded localization for generated chest X-ray reports. Below are concise definitions and the recommended evaluation protocol to reproduce the benchmark results reported for CXRReportGen.

Table 1: Key Evaluation Metrics for CXRReportGen Outputs

| Metric | What it Measures | Higher Score Meaning |
|---|---|---|
| Precision / Recall (P/R) (per finding) | Fraction of predicted findings that are correct (Precision) and fraction of true findings that are captured (Recall), often aggregated across all cases. Typically combined into F1 score for summary. | Higher precision = fewer false findings; higher recall = fewer missed findings. Both high = model outputs are accurate and comprehensive. |
| F1 Score (micro) | Combined measure of Precision and Recall. Micro-averaged F1 treats all findings equally across the dataset (suitable for multi-label evaluation). Often computed for specific label sets (e.g. CheXpert). | Closer to 100% means balanced high precision and recall – the model is identifying findings correctly and not missing many. |
| ROUGE-L | Overlap of sequences between generated and reference report (Longest Common Subsequence). Focuses on recall of content. | Higher = model covered more of the reference report’s content. |
| BLEU-4 | N-gram overlap between generated and reference text (up to 4-word phrases). Indicates fluency and phrasing similarity. | Higher = output text closely matches reference phrasing. |
| Box-Completion Precision | Correctness of predicted bounding boxes (what fraction correspond to true findings). | Higher = model’s highlighted regions on X-ray are usually relevant (few false highlights). |
| Box-Completion Recall | Completeness of predicted boxes (what fraction of true finding locations were detected). | Higher = model found most of the actual findings in the image (few misses). |

Evaluation Protocol (recommended)

1. Prepare Ground Truth: Collect a set of chest X-ray studies with expert-written reports. Two suitable datasets are commonly used: the MIMIC-CXR test set and the GR-Bench dataset. Ensure these reference reports are segmented if needed (e.g., identify the “Findings” section). If evaluating bounding boxes, ensure ground truth boxes are available for findings (as provided in GR-Bench).

2. Run CXRReportGen on the Test Set: Using the model in a controlled development environment (e.g., deployed on Azure ML or local inference), generate reports for each X-ray. Include both frontal and lateral images if the dataset provides both – the model accepts an optional lateral view input which can improve findings. Also provide any available textual context (indication, technique, etc.) if applicable, as the model input allows these fields.

3. Compute Text Similarity Metrics (BLEU, ROUGE): Use standard NLG evaluation scripts or libraries to compare each generated report with the reference report, computing BLEU-4 and ROUGE-L for each, then average over the test set. These metrics will give a general sense of how close the AI reports are to what radiologists wrote, in terms of wording and content overlap.

4. Compute Clinical Label Metrics (CheXpert F1): Use a CheXpert labeler or similar tool on both the generated and true reports to extract binary labels for a set of pathologies. Then compute precision, recall, F1 for each label and aggregate (micro-average) across all labels and samples. Pay attention to cases of uncertainty or negation in reports – ensure the labeling tool accounts for “no evidence of X” appropriately (the CheXpert labeler does). Report overall micro-F1 and optionally per-label performance to identify systematic weaknesses.

5. Compute RadGraph F1: Use the RadGraph evaluation script to analyze the generated and reference reports. Extract entities and relations from both reports and determine matches. The output is an F1 score that penalizes missing or hallucinated findings in the context of the report’s factual content.

6. Evaluate Bounding Box Alignment: For cases with ground truth bounding boxes (such as GR-Bench), evaluate each predicted box by calculating overlap (e.g. Intersection over Union). Determine matches with a threshold (e.g. IoU > 0.5). Calculate precision = (true positive boxes / predicted boxes) and recall = (true positive boxes / ground truth boxes) across the dataset. Visually inspect a sample of images with overlaid boxes to qualitatively assess correspondence between boxes and described findings.

7. Statistical Significance (optional): For comparisons (e.g., between models or after fine-tuning), consider bootstrap resampling or other statistical tests for metrics like BLEU or F1 to determine whether observed differences are meaningful.

The chosen metrics cover language fidelity, clinical accuracy, and visual localization — together providing a comprehensive picture of model performance. Follow the procedure above to quantify where CXRReportGen performs well and where it needs improvement before integration into downstream applications.

## Benchmark Results on Standard Datasets

Using the above metrics, CXRReportGen benchmark results reported on two test sets are summarized below.

Table 2: CXRReportGen Benchmark Performance

| Metric | MIMIC-CXR Test (Findings Generation) | GR-Bench Test (Grounded Reporting) |
|---|---:|---:|
| CheXpert F1-5 (micro) | 59.7 | – |
| ROUGE-L | 39.1 | 56.6 |
| BLEU-4 | 23.7 | – |
| Box-Completion Precision | – | 71.5% |
| Box-Completion Recall | – | 82.0% |

These benchmark numbers illustrate typical performance ranges and demonstrate how combining language and clinical-structure metrics yields a fuller evaluation of generated radiology reports.


## License

This repository is made publicly available under the MIT License.


## Citing

```bibtex
@article {Yu2022.08.30.22279318,
    author = {Yu, Feiyang and Endo, Mark and Krishnan, Rayan and Pan, Ian and Tsai, Andy and Reis, Eduardo Pontes and Fonseca, Eduardo Kaiser Ururahy Nunes and Ho Lee, Henrique Min and Abad, Zahra Shakeri Hossein and Ng, Andrew Y. and Langlotz, Curtis P. and Venugopal, Vasantha Kumar and Rajpurkar, Pranav},
    title = {Evaluating Progress in Automatic Chest X-Ray Radiology Report Generation},
    elocation-id = {2022.08.30.22279318},
    year = {2022},
    doi = {10.1101/2022.08.30.22279318},
    publisher = {Cold Spring Harbor Laboratory Press},
    URL = {https://www.medrxiv.org/content/early/2022/08/31/2022.08.30.22279318},
    eprint = {https://www.medrxiv.org/content/early/2022/08/31/2022.08.30.22279318.full.pdf},
    journal = {medRxiv}
}
```
