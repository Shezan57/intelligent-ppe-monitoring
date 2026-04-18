# Chapter 7: Conclusion and Future Work

## 7.1 Conclusion

This thesis presented the design, implementation, and evaluation of an Intelligent PPE Compliance Monitoring System for construction site safety. The work was motivated by three persistent problems in existing automated safety monitoring approaches: the inability to reliably detect the absence of protective equipment, the generation of excessive alerts that cause safety officer disengagement, and the computational impracticality of dense verification in real-time video streams.

To address these problems, the system introduced the Sentry-Judge architecture — a decoupled, asynchronous pipeline in which a fast YOLO-based detector handles real-time frame analysis at ~47 FPS, while an asynchronous SAM-based verifier processes uncertain Region of Interest crops in a background thread. A five-path intelligent triage mechanism routes 83.6% of detections through fast decision paths that require no SAM invocation, enabling the system to operate within the computational budget of a single edge GPU.

The key research contributions of this work are:

1. **A unified, 29,053-image multi-source PPE dataset** standardized to a five-class schema (`helmet`, `vest`, `person`, `no-helmet`, `no-vest`), assembled from seven construction site datasets.

2. **The five-path triage algorithm**, which classifies each detected person into one of five decision paths based on the presence or absence of PPE classes, minimizing SAM invocations by 83.6%.

3. **The Sentry-Judge asynchronous pipeline**, which achieves ~47 FPS real-time throughput while providing dense SAM verification for uncertain detections, directly resolving the absence detection paradox.

4. **The YOLOv11m model**, trained over 50 epochs to achieve 89.3% mAP@50 and 92.3% AP for the `no-helmet` class, with a 78% reduction in false positive rate after Judge verification.

5. **The Agentic Reporter module**, which generates OSHA-format daily compliance reports in PDF format, eliminating the need for continuous real-time alerts and addressing the alert fatigue problem.

6. **The Violation Chatbot**, which provides a text-to-SQL natural language interface for querying violation records, enabling non-technical site managers to retrieve safety statistics without SQL knowledge.

The system was evaluated through six experiments covering model comparison, per-class AP analysis, confusion matrix analysis, confidence threshold sensitivity, triage path distribution, and system latency profiling. Results confirmed that all six research objectives were met.


## 7.2 Future Work

Several directions are identified for extending this research:

### 7.2.1 Extended PPE Coverage

The current system detects only helmets and vests. Future work should extend detection to the full range of OSHA-mandated PPE including safety goggles, gloves, steel-toed boots, fall protection harnesses, and respiratory equipment. This requires curating or annotating appropriate training data, as existing public construction safety datasets provide limited coverage of these additional classes.

### 7.2.2 Multi-Camera Deployment

The current system architecture supports a single camera feed. A natural extension is a multi-camera deployment framework in which individual Sentry instances share a central database. This would enable zone-level compliance statistics, cross-zone worker tracking (if worker identification is feasible), and a unified site-wide dashboard.

### 7.2.3 Worker Identity and Re-identification

The current system assigns arbitrary track IDs to workers that do not persist across sessions or camera zones. Future work could integrate a person re-identification model [10][11] to provide persistent worker identities — enabling per-worker compliance histories and trend analysis over time. This would require careful consideration of privacy regulations and informed consent protocols.

### 7.2.4 Night-time and Adverse Weather Performance

The training dataset is biased toward daytime, clear-weather conditions. Future work should collect and annotate construction site imagery under nighttime, rain, fog, and dust conditions, and evaluate the system's robustness under these operating conditions. Image enhancement preprocessing (e.g., low-light enhancement via retinex-based methods) may partially mitigate this limitation.

### 7.2.5 Offline Chatbot Deployment

The current chatbot relies on the OpenAI API, incurring per-query costs and requiring internet connectivity. Future work should evaluate locally-hosted small language models (e.g., Phi-3, Mistral 7B, or LLaMA 3) as drop-in replacements for the OpenAI backend, enabling offline deployment while maintaining acceptable text-to-SQL accuracy.

### 7.2.6 Formal Evaluation of the Judge Module

The current evaluation infers SAM's verification accuracy from the reduction in downstream false positive rate. A more rigorous evaluation would collect pixel-level ground truth segmentation masks for a held-out set of ROI images, enabling direct measurement of mask coverage accuracy and the determination of an optimal coverage threshold per PPE class.


## 7.3 Final Remarks

Workplace safety and the protection of construction workers from preventable injuries represents both an ethical imperative and a legal obligation for construction organizations globally. Computer vision systems capable of automating PPE compliance monitoring at scale have the potential to meaningfully reduce workplace fatalities — a problem that, as noted in Chapter 1, claimed 1,075 lives in the U.S. construction sector alone in 2023 [25].

This work demonstrates that the technical barriers to reliable, real-time, absence-verified PPE monitoring are surmountable within the hardware constraints of a single-GPU edge device. The decoupled Sentry-Judge approach offers a practical template for balancing detection speed, verification accuracy, and computational cost — a trade-off that will remain relevant as both detection models and verification architectures continue to evolve.
