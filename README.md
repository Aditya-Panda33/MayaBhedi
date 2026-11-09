# Audio Deepfake Detection Pipeline using Wav2Vec 2.0

A complete, modular, and robust deep learning pipeline for detecting AI-generated synthetic speech (deepfakes) vs. authentic human voice using Hugging Face **Wav2Vec 2.0** (`facebook/wav2vec2-base`), PyTorch, and Librosa.

---

## 🏆 Test Set Performance (450 Held-Out Samples)

Evaluated on an unseen balanced test split (225 Real, 225 Fake):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.9844** | **98.44%** |
| **Precision** | **0.9823** | **98.23%** |
| **Recall** | **0.9867** | **98.67%** |
| **F1-Score** | **0.9845** | **98.45%** |
| **ROC-AUC** | **0.9979** | **99.79%** |

### Confusion Matrix Breakdown
- **True Real (0):** 221 / 225 correctly classified (**98.22%**)
- **True Fake (1):** 222 / 225 correctly classified (**98.67%**)

---

## 📂 Project Structure

All modules in `src/` are formatted as **100% self-contained Jupyter Notebooks (`.ipynb`)** with no external `.py` dependencies:

```
E:\Deepfake/
├── venv/                                    # Virtual environment (Python 3.13 + CUDA 12.4)
├── data/
│   ├── raw/                                 # Raw audio files (FLAC, WAV, M4A)
│   │   ├── FAKE/                            # AI-synthesized audio (25,436 files)
│   │   └── Real_audio/                      # Genuine human voice (32,406 files)
│   └── processed_clean/                     # Cleaned 16kHz Mono 4.0s Normalized WAVs
│       ├── manifest.csv                     # Audio metadata index
│       ├── train_split.csv                  # 70% Train split (2,100 samples)
│       ├── val_split.csv                    # 15% Validation split (450 samples)
│       └── test_split.csv                   # 15% Test split (450 samples)
├── graphs/                                  # EDA Visualizations (Step 3)
│   ├── 01_class_distribution.png
│   ├── 02_duration_distribution_before_after.png
│   ├── 03_waveform_comparison.png
│   └── 04_mel_spectrogram_comparison.png
├── evaluation/                              # Evaluation Reports & Figures (Step 5)
│   ├── confusion_matrix.png                 # Test Confusion Matrix heatmap
│   ├── roc_curve.png                        # ROC Curve (AUC = 0.9979)
│   ├── metrics_report.json                  # Quantitative metrics in JSON
│   └── metrics_report.txt                   # Detailed text evaluation summary
├── models/                                  # Exported Model Artifact (Step 6)
│   └── best_wav2vec2_deepfake_model.pth     # Serialized production model (360.85 MB)
├── logs/                                    # Detailed execution & audit logs
│   ├── corrupted_or_unreadable_files.json
│   ├── dataset_inspection_summary.json
│   ├── preprocessing_failures.json
│   └── training_history.json
├── src/                                     # Step-by-Step Modular Notebooks
│   ├── config.ipynb                         # Configurations, hyperparameters & paths
│   ├── explore_data.ipynb                   # Step 1: Data exploration & profiling
│   ├── preprocess.ipynb                     # Step 2: Resampling, Mono, 4.0s Equalization
│   ├── eda_visualize.ipynb                  # Step 3: Interactive EDA plotting
│   ├── dataset.ipynb                        # Step 4: Dataset & DataLoader inspection
│   ├── model.ipynb                          # Step 4: Wav2Vec2 + MLP architecture
│   ├── train.ipynb                          # Step 4: Mixed precision training loop
│   ├── evaluate.ipynb                       # Step 5: Test evaluation & confusion matrix
│   ├── export.ipynb                         # Step 6: Model serialization
│   └── inference.ipynb                      # Step 6: Standalone live audio prediction
├── tests/
│   └── test_pipeline.ipynb                  # Automated unit & integration test notebook
├── Audio_Deepfake_Detection_Pipeline.ipynb  # Comprehensive End-to-End Master Notebook
├── main.ipynb                               # Master execution notebook
├── web_app.ipynb                            # Interactive Browser Web Application
├── requirements.txt                         # Pinned project dependencies
├── .gitignore                               # Security filter (API keys, data, models)
└── README.md
```

---

## 🛠️ Installation & Environment Setup

### 1. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
# For CUDA GPU acceleration:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Connect VS Code Kernel
When opening any `.ipynb` notebook in VS Code:
1. Click **Select Kernel** at the top right.
2. Choose **Python Environment...**
3. Select **`venv (Python 3.13.5)`** or paste: `e:\Deepfake\venv\Scripts\python.exe`.

---

## 🗺️ Execution Roadmap

### Option A: The All-in-One Master Notebook
Open [**`Audio_Deepfake_Detection_Pipeline.ipynb`**](file:///e:/Deepfake/Audio_Deepfake_Detection_Pipeline.ipynb) and click **"Run All"**. It runs through every step sequentially with rich markdown, inline plots, metrics, and live audio prediction.

### Option B: Step-by-Step Modular Notebooks in `src/`
Run each stage in chronological order:
1. [**`src/explore_data.ipynb`**](file:///e:/Deepfake/src/explore_data.ipynb): Scans dataset, computes duration stats, identifies corrupted files.
2. [**`src/preprocess.ipynb`**](file:///e:/Deepfake/src/preprocess.ipynb): Standardizes audio to 16kHz Mono, 4.0s (64,000 samples), and peak normalization.
3. [**`src/eda_visualize.ipynb`**](file:///e:/Deepfake/src/eda_visualize.ipynb): Generates the 4 EDA graphs.
4. [**`src/train.ipynb`**](file:///e:/Deepfake/src/train.ipynb): Trains the Wav2Vec2 classifier with mixed precision on GPU.
5. [**`src/evaluate.ipynb`**](file:///e:/Deepfake/src/evaluate.ipynb): Evaluates the test set, plots Confusion Matrix and ROC Curve.
6. [**`src/inference.ipynb`**](file:///e:/Deepfake/src/inference.ipynb): Tests audio prediction on real vs. fake speech.

---

## 🎙️ Interactive Browser Web Application

Run [**`web_app.ipynb`**](file:///e:/Deepfake/web_app.ipynb) to launch a local browser web interface at:
`http://localhost:7860`

Features:
- Drag-and-drop or upload any `.wav` or `.flac` audio file.
- Built-in audio player with waveform playback.
- Pre-loaded buttons to test authentic human voice vs. AI synthetic deepfake.
- Real-time confidence score and probability breakdown.
- Fast inference latency ($\approx 45\text{ ms}$ on RTX 4060 GPU).

---

## 🔒 Security & Secrets Management
- All API keys, environment variables (`.env`, `HF_TOKEN`, tokens, credentials), raw audio archives, and large binary models are strictly ignored via [**`.gitignore`**](file:///e:/Deepfake/.gitignore) to prevent accidental repository commits.
