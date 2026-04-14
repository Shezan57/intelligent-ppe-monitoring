# Chapter 1: Introduction

## 1.1 Background

Construction remains one of the most hazardous industries worldwide. According to the U.S. Bureau of Labor Statistics, the construction sector recorded 1,075 workplace fatalities in 2023, accounting for approximately 20% of all occupational deaths [25]. Falls, struck-by incidents, and exposure to harmful environments constitute the leading causes of these fatalities, many of which are preventable through the consistent use of Personal Protective Equipment (PPE) such as hard hats and high-visibility vests [26].

Despite regulatory mandates from organizations such as the Occupational Safety and Health Administration (OSHA), PPE compliance on active construction sites remains inconsistent. Manual monitoring by safety officers is labor-intensive, subjective, and limited in coverage — a single inspector cannot continuously observe every worker across a multi-zone site. This gap between safety requirements and enforcement capabilities motivates the development of automated, vision-based compliance monitoring systems.

Recent advances in deep learning-based object detection, particularly the YOLO (You Only Look Once) family of architectures [1][5][6][7], have demonstrated that real-time detection of PPE items is technically feasible. However, deploying these systems in practice reveals a critical limitation: while object detectors reliably identify the *presence* of equipment (e.g., a helmet on a worker's head), they exhibit significantly lower precision when confirming the *absence* of equipment — a condition this work terms the "absence detection paradox." A worker without a helmet does not produce a distinctive visual feature; rather, the system must infer a negative from an unbounded set of possible head appearances.


## 1.2 Problem Statement

Existing computer vision approaches to PPE monitoring suffer from three interconnected problems:

1. **The Absence Detection Paradox.** Single-stage detectors such as YOLO achieve strong precision for positive PPE classes (helmet present) but produce unacceptable false positive rates when detecting negative classes (helmet absent). Preliminary experiments in this work measured precision as low as 41% for the `no-helmet` class using a standard YOLO detector alone.

2. **Alert Fatigue.** Systems that issue real-time notifications for every detected violation generate an overwhelming volume of alerts — many of which are false positives. Site managers routinely disable or ignore such systems, rendering them ineffective [25].

3. **Computational Constraints.** Dense semantic segmentation models such as SAM [8] can verify PPE absence with high accuracy but operate at less than one frame per second, making direct integration into a live video pipeline impractical for real-time monitoring.

No existing system adequately addresses all three problems simultaneously. A solution is required that combines the speed of single-stage detection with the accuracy of semantic verification, while presenting results in a format that eliminates alert fatigue.


## 1.3 Research Objectives

This work aims to design, implement, and evaluate an Intelligent PPE Compliance Monitoring System that addresses the problems identified above. The specific objectives are:

1. To construct a large-scale, unified dataset suitable for training a multi-class PPE detector with both presence and absence classes.

2. To design a decoupled, asynchronous pipeline architecture (the "Sentry-Judge" system) that achieves real-time detection speeds while maintaining high verification accuracy.

3. To develop a five-path intelligent triage mechanism that minimizes the computational overhead of semantic verification by routing only uncertain detections to the verification stage.

4. To implement an automated, LLM-based reporting module that generates actionable daily compliance summaries, eliminating the need for continuous real-time alerts.

5. To develop a natural language chatbot interface that enables site managers to query violation data conversationally.

6. To evaluate the system through empirical experiments comparing model architectures, tracking algorithms, and detection confidence thresholds.


## 1.4 Scope and Limitations

This work focuses on the detection of two PPE items: hard hats (helmets) and high-visibility vests. While construction sites require additional equipment such as safety goggles, gloves, and steel-toed boots, the detection of these smaller or partially occluded items presents distinct challenges that are beyond the scope of this study.

The system is designed for deployment on a single edge device equipped with a mid-tier GPU (e.g., NVIDIA T4). Multi-site, cloud-based deployments and cross-camera worker re-identification are not addressed.

The chatbot module relies on a third-party API (OpenAI GPT-4o-mini) for natural language processing via text-to-SQL conversion, which introduces a dependency on external infrastructure and associated costs.


## 1.5 Thesis Organization

The remainder of this thesis is organized as follows:

- **Chapter 2** reviews related work in object detection, semantic segmentation, multi-object tracking, and automated safety monitoring systems.
- **Chapter 3** presents the system architecture, including the Sentry-Judge pipeline, five-path triage logic, and chatbot design.
- **Chapter 4** describes the implementation details, including dataset engineering, model training, and software development.
- **Chapter 5** presents the experimental results, including ablation studies, performance comparisons, and system evaluation.
- **Chapter 6** discusses the findings, limitations, and practical implications of the work.
- **Chapter 7** concludes the thesis and identifies directions for future research.
