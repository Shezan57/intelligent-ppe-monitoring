# Thesis Progress & Analysis Report

Based on the development of the Intelligent PPE Compliance Monitoring System and its architectural evolution, here are the formal responses to your thesis submission requirements.

## 1. Phased Results (What has been achieved so far)

**Phase 1: Dataset Engineering & Baseline Modeling**
- Successfully merged 6 disparate datasets into a unified 29,053-image dataset with a standardized 5-class schema (`helmet`, `vest`, `person`, `no-helmet`, `no-vest`).
- Implemented data sanitization pipelines, explicitly removing duplicated frames (111 images) and filtering bulk out-of-domain edge cases.
- Trained a robust YOLO26m (and experimented with YOLO11m) baseline model that achieved a strong overall **mAP@50 metric of ~89.7%**.

**Phase 2: Architectural Shift - The "Sentry-Judge" Safety System**
- Identified that standard object detectors excel at presence detection but struggle with absence detection (~41% precision for missing PPE).
- Transitioned the core infrastructure into a **decoupled, asynchronous pipeline** called the "Sentry-Judge" system to resolve real-time processing constraints:
  - **Module 1 (The Sentry):** Uses YOLO combined with high-speed multi-object tracking (e.g., ByteTrack/BoT-SORT mechanics) acting as a real-time producer (running at ~30 FPS). It applies a **5-path triage logic** to pre-filter false alarms and incorporates a strict **5-minute cooldown timer** per unique worker ID to eliminate violation spamming.
  - **Module 2 & 3 (The Judge):** Acts as an asynchronous consumer using SAM 3 (Segment Anything Model) to deeply verify only the cropped Head/Torso ROI of suspected violations passed by the Sentry. Verified violations are officially logged into an SQLite database.
  - **Module 4 (The Agentic Reporter):** An end-of-day LLM agent aggregates verified, deduplicated violations from the database to generate an actionable, professional daily OSHA summary PDF.

**Phase 3: Model Refinement & Dataset Limitations**
- Conducted fine-tuning experiments adding 2,550 person-specific images to target the baseline's weakest class (`person`), discovering that augmenting with non-construction stock photography causes a domain mismatch (dropping `person` AP from 85.3% to 83.1%). This proved the risk of naive data augmentation without domain alignment.
- Nonetheless, the combined training effectively pushed the safety-critical `no-helmet` detection AP to **91.8%**.

---

## 2. There are problems (Current challenges)

**1. Queue Flooding vs. Cooldown Mechanics**
Before introducing the tracker and cooldown logic, the Sentry outputting bounding boxes at 30 FPS would flood the SAM "Judge" queue with up to 30 requests per second for a single worker missing a helmet. While the new 5-minute cooldown mechanically solves queue overload, cross-camera re-identification (maintaining worker IDs across different camera zones) remains extremely challenging. If a worker moves to a new camera feed, tracking ID clears, potentially causing duplicate violation logging.

**2. Validation Metrics vs. Real-World Generalization**
During trials, the baseline model achieved an impressive 89.7% mAP@50 on paper, while the model trained on the combined dataset (baseline + 2,550 external person images) showed slightly lower formal metrics (89.3%). However, real-world deployment tests proved the inverse: the combined model correctly identified ambiguous edge cases at a 25% confidence threshold, whereas the baseline model struggled even at 10%. This highlights a critical finding for computer vision deployments: **optimizing blindly for validation mAP often leads to overfitting to the dataset distribution**, whereas injecting varied (even slightly out-of-domain) data forces the neural network to generalize better, yielding superior real-world robustness despite lower theoretical scores.

**3. The Absence Detection Paradox in Real-time Constraints**
Relying solely on single-stage detectors like YOLO to confirm the *absence* of gear (like a helmet) typically yields high false negatives due to complex backgrounds. Using a heavy semantic model like SAM to verify absence solves the precision issue but introduces a massive asynchronous bottleneck (SAM runs natively at < 1 FPS). Thus, balancing the Sentry's queue payload against the Judge's processing limits presents an ongoing system-engineering optimization problem.

---

## 3. Research methods and feasibility analysis to be adopted

### Research Methods
**1. Decoupled Asynchronous Semantic Verification**
The research methodology adopts a strict microservices approach: YOLO serves as an ultra-fast synchronous Region Proposal Network (The Sentry). When specific conditions are met (Paths 1, 2, 3, 4), the Sentry crops the exact ROI, caches it locally, drops an IPC message into a queue, and immediately continues tracking. The SAM model (The Judge) asynchronously consumes these ROI messages, structurally isolating the real-time video pipeline from the heavy inference overhead.

**2. Empirical Ablation Studies**
To scientifically justify the framework, comparative empirical analyses will be conducted:
- Comparing YOLO object detectors (YOLO11m vs YOLO26m) for the Sentry module.
- Testing the tracking efficiency of custom IoU vs standard ByteTrack/BoT-SORT modules inside the Sentry loop.

### Feasibility Analysis

**1. Technical & Performance Feasibility (Proven)**
The introduction of the "cooldown timer" attached to unique tracking IDs makes the application technically feasible. Instead of evaluating 30 objects a second per person, the Sentry evaluates a person once and ignores them for 5 minutes. Coupled with the 5-path triage filtering heavily obvious safe cases, SAM's workload is functionally minimized, ensuring the main process sustains 30 FPS.

**2. Practical Reporting Feasibility**
The framework overcomes the primary fatal flaw of computer vision in safety management: "Alert Fatigue". Because the LLM "Agentic Reporter" only acts on deduplicated, Judge-verified data at the end of the day, site managers receive one actionable PDF rather than thousands of meaningless SMS alerts, proving extremely high real-world use-case feasibility.

**3. Economic & Deployment Feasibility**
By decoupling the architecture, the heavy SAM model doesn't need scaling equal to the video influx. The tracking queue prevents back-pressure limits, meaning this robust safety suite can be viably deployed on mid-tier hardware (e.g., single edge devices utilizing a T4 equivalent) rather than requiring massive enterprise server clusters.
