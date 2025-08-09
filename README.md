# Automated Product-Price Matching Pipeline

A high-performance computer vision and graph-based solution to automate product–price association from retail shelf images, achieving over **92% accuracy** in matching.

---

## 📌 Project Overview

CPG distributors face substantial costs ensuring product displays at retail locations match planned layouts and pricing. Manual validation is labor-intensive and error-prone. This project delivers an **automated pipeline** leveraging AI to streamline execution compliance and deliver scalable insights for distributors.

---

## 🔹 Key Components

| Stage                  | Technique & Tools Used                           | Outcome / Impact                            |
|------------------------|---------------------------------------------------|----------------------------------------------|
| **Object Detection**   | YOLOv11 fine-tuned on 6K+ images (342K+ bounding boxes) | Accurate detection of products & price tags |
| **Price OCR**          | Multimodal LLMs (LLaVA via Segmind API)           | Parsed price text from cropped tag regions   |
| **Association**        | Graph Neural Network (PyTorch Geometric)          | Linked products to correct price tags (92%+ accuracy) |

---

## 🚀 Features & Benefits

- **Automated Execution Compliance**: Reduces manual verification by accurately matching displays to “ground truth.”
- **Scalable Processing**: Built for large datasets using optimized preprocessing and imbalance handling.
- **End-to-End Pipeline**: Covers detection, extraction, and graph-based association — ideal for retail analytics applications.

---

## 🛠 Tech Stack

- **Detection**: YOLOv11 (Ultralytics)
- **OCR**: LLaVA + Segmind API
- **Graph Learning**: Graph Neural Network (PyTorch Geometric)
- **Programming Language**: Python
- **Libraries & Tools**: NumPy, Pandas, OpenCV, Torch, Transformers
- **Hardware**: NVIDIA GeForce RTX 3070 GPUs

---

## 📊 Results

- **YOLO mAP@50**: > 80%
- **Association Accuracy**: 92%+
- **GNN Precision**: ~89.7%
- **GNN Recall**: ~97.4%
- Reduced manual verification needs, enabling scalable execution compliance analysis for CPG distributors

---

## 🔮 Future Directions

- Fine-tune LLaVA with a dedicated price-tag dataset to improve OCR performance.
- Integrate all components into a unified, end-to-end automated pipeline.
- Expand labeled datasets to improve generalization across CPG categories.
- Explore heterogeneous GNNs for richer node/edge representation.

---

##  Authors & Acknowledgments

- **Tyler Thompson** — Box Detection  
- **Salsabil Arabi** — Price Extraction  
- **Stockton Jenkins** — Graph Association  
Dataset provided by **Delicious AI**.
