# Chapter 2: Literature Review

## 2.1 Introduction

This chapter reviews the existing body of research across the four technical domains that underpin the proposed system: object detection architectures, semantic segmentation models, multi-object tracking algorithms, and automated PPE compliance monitoring systems. The review establishes the theoretical foundations for the design decisions presented in Chapter 3 and identifies the specific gaps that this work addresses.


## 2.2 Object Detection Architectures

### 2.2.1 Evolution of the YOLO Family

The You Only Look Once (YOLO) framework, introduced by Redmon et al. in 2016 [1], fundamentally changed the object detection landscape by framing detection as a single-pass regression problem rather than a multi-stage pipeline. By dividing the input image into a grid and predicting bounding boxes and class probabilities simultaneously, YOLOv1 achieved real-time inference speeds that were previously unattainable.

Subsequent iterations improved upon this foundation. YOLOv3 [2] introduced multi-scale detection using Feature Pyramid Networks (FPN) [19], enabling the model to detect objects at different scales within the same image. YOLOv4 [3] incorporated a "bag of freebies" — training-time augmentation strategies that improved accuracy without increasing inference cost.

YOLOv5 [4], released by Ultralytics in 2020, transitioned the YOLO ecosystem from the original Darknet framework to PyTorch, significantly expanding accessibility and community adoption. Although no formal research paper accompanied its release, YOLOv5 became the most widely deployed YOLO variant in industrial applications due to its robust training pipeline and extensive documentation.

YOLOv7 [5], published at CVPR 2023 by Wang et al., introduced trainable bag-of-freebies techniques and extended efficient layer aggregation networks (E-ELAN), achieving state-of-the-art accuracy on the MS COCO benchmark while maintaining real-time inference speeds.

YOLOv8 [6], released by Ultralytics in January 2023, adopted an anchor-free detection head and a decoupled head architecture that separates classification and localization tasks. This architectural change improved both accuracy and training convergence speed.

The model selected for this work, YOLO26m [7], represents the latest generation in the Ultralytics YOLO family at the time of development. It extends the prior architecture with improved feature extraction, enhanced attention mechanisms, and optimized model scaling, providing a practical balance between accuracy and inference speed for the construction site monitoring domain.

### 2.2.2 Attention Mechanisms in Detection

Attention mechanisms have emerged as a key technique for improving detection accuracy, particularly for small or partially occluded objects. The Convolutional Block Attention Module (CBAM) [20] applies sequential channel and spatial attention to feature maps, enabling the network to focus on relevant regions while suppressing irrelevant background features. The Transformer-based self-attention mechanism [21], originally proposed for natural language processing, has been adapted for computer vision tasks, forming the basis for detection architectures such as DETR and its variants.

In the context of PPE detection, attention mechanisms are particularly valuable for identifying small objects such as helmets at distance or partially visible vests behind construction equipment.


## 2.3 Semantic Segmentation

### 2.3.1 The Segment Anything Model (SAM)

The Segment Anything Model (SAM), introduced by Kirillov et al. at ICCV 2023 [8], represents a paradigm shift in image segmentation. Trained on a dataset of over 1 billion masks across 11 million images, SAM demonstrates zero-shot segmentation capabilities: it can generate high-quality segmentation masks for arbitrary objects without task-specific fine-tuning.

SAM's architecture comprises three components: an image encoder (a Vision Transformer), a prompt encoder (accepting points, boxes, or text as input), and a lightweight mask decoder. The image encoder produces a high-dimensional feature embedding of the input image, which the mask decoder uses — conditioned on the prompt — to generate one or more segmentation masks.

For this work, SAM serves as the verification backbone (the "Judge") that confirms or denies the presence of PPE items within cropped regions of interest. Its zero-shot capability is particularly valuable because it eliminates the need for PPE-specific segmentation training data.

### 2.3.2 Mask R-CNN

Prior to SAM, Mask R-CNN [9] represented the standard approach for instance segmentation. Built on the Faster R-CNN detection framework, Mask R-CNN adds a parallel branch that predicts a pixel-level segmentation mask for each detected object. While Mask R-CNN achieves high accuracy on established benchmarks, it requires class-specific training and does not generalize to unseen object categories. SAM's zero-shot capability provides a significant advantage for applications where training data for specific PPE configurations is limited.


## 2.4 Multi-Object Tracking

### 2.4.1 Tracking-by-Detection Paradigm

Modern multi-object tracking (MOT) algorithms follow the tracking-by-detection paradigm, where objects are first detected in individual frames and then associated across frames to form continuous trajectories. The association step is typically solved using a combination of motion prediction (Kalman filtering) and appearance similarity metrics.

SORT (Simple Online and Realtime Tracking) [12] established the foundation for this approach by combining a Kalman filter for motion estimation with the Hungarian algorithm for bounding box association based on Intersection over Union (IoU). SORT's simplicity enables real-time operation but suffers from identity switches when objects undergo occlusion.

The IoU metric, central to both object detection evaluation and tracking association, is defined as the ratio of the area of intersection to the area of union between two bounding boxes $A$ and $B$:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|} \tag{2.1}
$$

An IoU of 1.0 indicates perfect overlap, while an IoU of 0.0 indicates no overlap. In tracking, two bounding boxes are associated as the same object if their IoU exceeds a configured threshold $\tau_{\text{IoU}}$ (set to 0.30 in this work).

### 2.4.2 ByteTrack

ByteTrack [10], proposed by Zhang et al. at ECCV 2022, addresses a key limitation of prior trackers: the loss of low-confidence detections. Traditional approaches discard detections below a confidence threshold before association. ByteTrack instead retains all detection boxes — including low-confidence ones — and performs two-stage association: high-confidence detections are matched first, followed by low-confidence detections. This approach significantly reduces missed detections and identity switches in crowded scenes.

### 2.4.3 BoT-SORT

BoT-SORT [11], proposed by Aharon et al. in 2022, extends the ByteTrack framework by incorporating camera motion compensation and a robust re-identification feature extractor. By explicitly modeling camera ego-motion, BoT-SORT reduces false associations caused by camera panning or vibration — a common occurrence in construction site surveillance where cameras may be mounted on moving equipment or temporary structures.

### 2.4.4 Custom IoU Tracking for This Work

For the specific requirements of construction site monitoring, this work employs a custom IoU-based tracker rather than adopting ByteTrack or BoT-SORT directly. The rationale is threefold: (1) construction workers exhibit limited mobility within camera zones, reducing the need for sophisticated motion models; (2) the cooldown mechanism (Section 3.3.2) requires tight integration between the tracker and the triage logic; and (3) eliminating the re-identification embedding network reduces computational overhead, contributing to the system's edge-device deployment target.


## 2.5 PPE Detection Systems

### 2.5.1 Existing Approaches

The application of deep learning to construction site PPE detection has attracted significant research attention. Fang et al. [13] demonstrated that convolutional neural networks can detect hardhats in construction images with accuracy exceeding 95% under controlled conditions. However, their evaluation was limited to clear, well-lit images and did not address real-time video processing.

Wu et al. [15] constructed an early benchmark dataset for hardhat detection and evaluated several detection architectures, concluding that single-stage detectors provide the best speed-accuracy tradeoff for real-time applications. Nath et al. [16] extended this work to include vest detection and proposed a multi-class framework, though their system processed images at only 5-8 FPS due to the computational cost of the selected architecture.

Fang et al. [14] addressed the detection of both workers and heavy equipment on construction sites, demonstrating the feasibility of multi-category detection using a unified model. Delhi et al. [17] provided a comprehensive review of PPE detection methods, identifying dataset quality, class imbalance, and real-time processing as the three primary challenges facing the field.

### 2.5.2 Limitations of Existing Work

The reviewed literature reveals three consistent limitations:

1. **No absence detection.** Existing systems detect only the presence of PPE items. None employ a verification mechanism for confirming that PPE is genuinely missing, leading to high false positive rates when a helmet or vest is simply occluded or at an unusual angle.

2. **No alert management.** Systems that issue immediate alerts for every detected violation do not address the alert fatigue problem that renders such systems impractical in real deployments.

3. **No conversational interface.** All reviewed systems provide dashboard-style visualization but lack natural language query capabilities for non-technical site managers.

This work addresses all three limitations through the Sentry-Judge architecture (absence verification), the Agentic Reporter (alert consolidation), and the Violation Chatbot (natural language queries).


## 2.5a Evaluation Metrics: Formal Definitions

The performance of object detection systems is evaluated using standard metrics derived from the confusion matrix. Let TP, FP, and FN denote the number of True Positives, False Positives, and False Negatives respectively at a given confidence threshold. The primary metrics used throughout this thesis are defined as follows.

**Precision** measures the proportion of positive detections that are correct:

$$
\text{Precision} = \frac{TP}{TP + FP} \tag{2.2}
$$

**Recall** (also called Sensitivity) measures the proportion of actual positive objects that are detected:

$$
\text{Recall} = \frac{TP}{TP + FN} \tag{2.3}
$$

**F1 Score** is the harmonic mean of Precision and Recall, providing a single balanced measure:

$$
F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \tag{2.4}
$$

**Average Precision (AP)** for a single class is computed as the area under the Precision-Recall (PR) curve, approximated by the interpolated 11-point average:

$$
AP = \int_0^1 p(r)\, dr \approx \frac{1}{11} \sum_{r \in \{0, 0.1, \ldots, 1.0\}} p_{\text{interp}}(r) \tag{2.5}
$$

where $p_{\text{interp}}(r) = \max_{\tilde{r} \geq r} p(\tilde{r})$ is the maximum precision at any recall $\tilde{r} \geq r$.

**mean Average Precision (mAP@50)** averages the AP across all $C$ detection classes evaluated at an IoU threshold of 0.50:

$$
\text{mAP}\text{@50} = \frac{1}{C} \sum_{c=1}^{C} AP_c \tag{2.6}
$$

For this work, $C = 5$ (helmet, vest, person, no-helmet, no-vest).


## 2.6 Large Language Models and Text-to-SQL

The integration of Large Language Models (LLMs) into database query interfaces represents an emerging research direction. OpenAI's GPT-4 [22] and its variants have demonstrated strong performance on text-to-SQL benchmarks, where natural language questions are translated into executable SQL queries.

Pourreza and Rafiei [23] proposed DIN-SQL, a decomposed approach that breaks the text-to-SQL task into sub-problems (schema linking, query classification, and SQL generation), achieving state-of-the-art accuracy on the Spider benchmark. Sun et al. [24] introduced SQL-PaLM, demonstrating that careful prompt engineering with few-shot examples and execution-based refinement significantly improves the reliability of LLM-generated SQL.

This work applies the text-to-SQL paradigm to the construction safety domain, enabling site managers to query violation records through a conversational interface without requiring SQL knowledge.


## 2.7 Summary of Literature Gaps

Table 2.1 summarizes the capabilities of existing PPE detection systems against the requirements addressed by this work.

**Table 2.1: Comparison of PPE Detection System Capabilities**

| Capability | Fang et al. [13] | Wu et al. [15] | Nath et al. [16] | Delhi et al. [17] | This Work |
|-----------|:-:|:-:|:-:|:-:|:-:|
| Real-time detection (>25 FPS) | ✗ | ✗ | ✗ | — | ✓ |
| Absence verification | ✗ | ✗ | ✗ | ✗ | ✓ |
| Alert fatigue mitigation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Daily report generation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Natural language query | ✗ | ✗ | ✗ | ✗ | ✓ |
| Edge-device viable | ✗ | ✓ | ✗ | — | ✓ |

This comparison establishes that no existing system simultaneously provides real-time detection, absence verification, alert management, and conversational querying — the combination that this work delivers.
