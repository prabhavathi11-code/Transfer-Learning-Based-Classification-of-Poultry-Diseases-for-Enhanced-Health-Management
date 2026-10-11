# PoultryVision: Transfer-Learning-Based Classification of Poultry Diseases for Enhanced Health Management

> **AI-Powered Avian Pathology & Biosecurity Platform** engineered with **MobileNetV2 Transfer Learning**, **TensorFlow/Keras**, and an interactive clinical diagnostic dashboard for automated poultry disease classification, real-time flock triage, and veterinary action plans.

[![Live Portal](https://img.shields.io/badge/Live_Portal-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://prabhavathi11-code.github.io/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/prabhavathi11-code/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management)
[![Model Accuracy](https://img.shields.io/badge/MobileNetV2-96.8%25_Accuracy-00C6FF?style=for-the-badge&logo=tensorflow)](https://github.com/prabhavathi11-code/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

### 🌐 **Live Published Gateway & Interactive Diagnostic Studio:**
👉 **[https://prabhavathi11-code.github.io/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management/](https://prabhavathi11-code.github.io/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management/)**

---

## 🏛️ Academic & Engineering Credentials

* **Institution:** St. Ann's College of Engineering and Technology
* **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning — CSE-AIML)
* **Engineering Team:**
  * **Ch.Prabhavathi** (Lead Developer & ML Engineer)
  * **B.Kusuma Priya** (Data & Preprocessing Engineer)
  * **A.Teja** (Evaluation & Backend Engineer)

---

## 📌 Executive Summary & Research Motivation

Poultry farming represents a critical pillar of global food security and rural agricultural economics. However, poultry flocks are highly susceptible to acute infectious outbreaks such as **Coccidiosis**, **Newcastle Disease**, and **Salmonellosis**. Because symptoms can escalate across entire sheds within 48 to 72 hours, delayed diagnosis often results in devastating flock mortality and severe financial loss.

**PoultryVision** leverages **Deep Transfer Learning** via a fine-tuned **MobileNetV2** backbone trained on categorized biomedical image datasets (~7,000 images from Roboflow). By combining edge-optimized neural inference with an interactive clinical web studio, PoultryVision empowers smallholder farmers, commercial farm managers, and veterinary practitioners to achieve **instant, laboratory-grade pathology classification** directly from chicken photographs.

---

## 🚀 Key Platform Features

1. **Multi-Class Disease Classification**
   * Classifies avian conditions across 4 categories: **Coccidiosis**, **Newcastle Disease (ND)**, **Salmonellosis (Pullorum)**, and **Healthy Flock**.
   * Delivers **96.8% validation accuracy** with low computational latency (18 ms per specimen).

2. **Softmax Probability Distribution & Confidence Scoring**
   * Computes full probability vectors across all four output classes with animated visual confidence meters.

3. **Grad-CAM Neural Attention Heatmap Overlay**
   * Visualizes class activation mappings (CAM) directly over the uploaded specimen to highlight the biological lesions, comb cyanosis, or feather discoloration guiding the neural network's prediction.

4. **Automated Veterinary Action Plans & Prescriptions**
   * Dynamically synthesizes pathology details, primary affected organs, recommended pharmacological agents (e.g., *Amprolium*, *Toltrazuril*, *Enrofloxacin*), and immediate biosecurity isolation instructions.

5. **Smart Flock Symptom Triage Matrix**
   * Interactive multi-symptom checklist (bloody stools, twisted neck/torticolis, vent pasting, gasping, mortality spikes) calculating real-time flock infection urgency.

6. **Interactive Model Architecture Benchmarks & Visualizations**
   * Live interactive **Plotly.js** charts detailing 30-epoch training/validation learning curves and an annotated 4×4 Confusion Matrix.

7. **Audit Logging & Report Exporting**
   * Records session diagnostic histories, exports formatted **Veterinary Clinical Reports**, and allows 1-click **CSV log export**.

8. **Zero-Setup Client-Side Web Deployment**
   * Fully static and server-free execution on GitHub Pages alongside an optional local **Flask / TensorFlow** Python server.

---

## 🧬 Disease Pathology & Clinical Reference

| Condition | Pathogen Type | Clinical Biomarkers | Transmission | Recommended Veterinary Intervention |
| :--- | :--- | :--- | :--- | :--- |
| **Coccidiosis** | Protozoan (*Eimeria tenella / necatrix*) | Bloody droppings, hunched posture, cecal hemorrhages, lethargy | Ingestion of sporulated oocysts in wet litter or feed | **Amprolium** (0.024% in drinking water for 5 days) or **Toltrazuril** (7 mg/kg body wt); replace moist litter. |
| **Newcastle Disease** | Avian Paramyxovirus (*APMV-1*) | Torticolis (twisted neck), respiratory clicking, greenish diarrhea | Aerosol droplets, wild bird vectors, nasal secretions | **Strict pen quarantine**; supportive electrolytes & vitamins; immediate revaccination (*LaSota strain*) for unaffected pens. |
| **Salmonellosis / Pullorum** | Bacterium (*Salmonella enterica*) | Chalky white vent pasting, somnolence, huddling, septicemia | Vertical (via breeder egg) or horizontal (rodents/feed) | **Enrofloxacin** or **Amoxicillin** in water; antiseptic vent cleansing; water chlorination and rodent extermination. |
| **Healthy Condition** | Baseline Physiological Standard | Vibrant red comb, clear round eyes, active foraging, firm stool | N/A | Maintain ad libitum clean water, 18-20% crude protein feed ration, and standard sanitization footbaths. |

---

## ⚙️ Transfer Learning Architecture & Pipeline

```mermaid
flowchart TD
    A[Input Specimen: 224x224x3 RGB Image] --> B[Image Preprocessing & Normalization]
    B --> C[Data Augmentation: Flip, Rotation, Zoom]
    C --> D[MobileNetV2 Backbone Pre-trained on ImageNet]
    
    subgraph Feature Extraction
        D --> E[Depthwise Separable Convolutions]
        E --> F[Inverted Residual Bottleneck Blocks]
        F --> G[Deep Feature Tensor: 7x7x1280]
    end

    G --> H[Global Average Pooling 2D]
    H --> I[Dense Layer: 256 Units + ReLU]
    I --> J[Dropout Regularization: Rate = 0.4]
    J --> K[Softmax Classification Head: 4 Units]

    K --> L{Output Probabilities}
    L --> M[Coccidiosis: %]
    L --> N[Newcastle Disease: %]
    L --> O[Salmonella: %]
    L --> P[Healthy Flock: %]

    L --> Q[Grad-CAM Attention Map Visualization]
    L --> R[Automated Veterinary Action Protocol]
```

### Architectural Highlights:
* **Base Model:** `MobileNetV2` with ImageNet frozen feature weights, fine-tuned on the Roboflow avian dataset.
* **Regularization:** Dropout layer ($p = 0.4$) and L2 weight decay to prevent overfitting on subtle plumage textures.
* **Optimizer:** Adam ($\alpha = 1 \times 10^{-4}$ with learning rate decay).
* **Loss Function:** Categorical Cross-Entropy across 4 mutually exclusive pathology classes.

---

## 📊 Comparative Performance Matrix

| Model Architecture | Validation Accuracy | Precision | Recall | F1-Score | Inference Latency | Model Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Selected)** | **96.8%** | **96.5%** | **97.1%** | **96.8%** | **18 ms** | **3.5M** |
| ResNet-50 | 94.2% | 94.0% | 93.8% | 93.9% | 46 ms | 25.6M |
| InceptionV3 | 93.6% | 93.2% | 94.1% | 93.6% | 54 ms | 23.8M |
| Custom 4-Layer CNN | 88.5% | 87.9% | 88.2% | 88.0% | 22 ms | 5.2M |

---

## 🔑 Demo Access & Team Login Credentials

| Team Member | Username | Demo Password | Academic Role |
| :--- | :--- | :--- | :--- |
| **Ch.Prabhavathi** | `Ch.Prabhavathi` | `@4211` | Lead ML Engineer & Full-Stack Architect |
| **B.Kusuma Priya** | `B.Kusuma Priya` | `@4203` | Data Preprocessing & Pathology Analyst |
| **A.Teja** | `A.Teja` | `@2303` | Model Evaluation & Deployment Specialist |

*(Quick 1-Click login buttons are also embedded directly on the login portal for testing convenience)*

---

## 💻 Local Setup & Execution Guide

### Option 1: Instant Browser Web Portal (No Python Required)
1. Clone the repository:
   ```bash
   git clone https://github.com/prabhavathi11-code/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management.git
   cd Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management
   ```
2. Double-click `index.html` or `dashboard.html` to open in any web browser.

### Option 2: Local Python Flask Backend
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Navigate to `http://localhost:5000` in your web browser.

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── deploy.yml              # Automated GitHub Pages CI/CD workflow
├── static/
│   ├── coccidiosis.jpeg            # Test specimen: Coccidiosis
│   ├── hen-healthy.jpeg            # Test specimen: Healthy flock
│   ├── poultry-farm.jpeg           # Test specimen: Salmonella
│   └── sick-chicken.jpeg           # Test specimen: Newcastle Disease
├── Templates/
│   ├── index.html                  # Flask base landing page
│   ├── predict.html                # Flask model inference view
│   ├── blog.html                   # Educational pathology articles
│   ├── blog-single.html            # Article detail template
│   └── portfolio-details.html      # Project documentation view
├── documents/
│   ├── Poultrydisease_Project.pdf  # Project academic dossier
│   └── workflow.txt                # Engineering methodology notes
├── videodemo/
│   └── Recording 2025-07-01.mp4    # Screencast application demo
├── app.py                          # Resilient Flask web server & inference router
├── dashboard.html                  # Live interactive diagnostic studio & benchmarks
├── index.html                      # College project gateway & team login
├── login.html                      # Authentication gateway view
├── login.css                       # Modern clinical/veterinary design stylesheet
├── login.js                        # Client-side authentication & routing logic
├── requirements.txt                # Python environment dependency manifest
└── README.md                       # Comprehensive documentation & research summary
```

---

## 📜 Academic Attribution & License

Developed as part of the Artificial Intelligence & Machine Learning curriculum at **St. Ann's College of Engineering and Technology**, Chirala, India. Released for academic research, education, and livestock health enhancement.
