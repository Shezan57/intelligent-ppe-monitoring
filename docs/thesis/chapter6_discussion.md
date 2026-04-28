# Chapter 6: Discussion

## 6.1 Introduction

This chapter interprets the experimental findings presented in Chapter 5, discusses  the system's contributions relative to existing work, examines its limitations, and considers broader implications for automated construction site safety monitoring.


## 6.2 Analysis of the Sentry-Judge Architecture

The core architectural contribution of this work is the decoupled Sentry-Judge pipeline, and the experimental results substantiate its design rationale. The Sentry's five-path triage mechanism proved highly effective in practice: the high-compliance test video achieved a 100% SAM bypass rate (Table 5.9), with all detections resolved via Path 1 (Fast Safe) without SAM invocation. While this extreme bypass rate reflects the specific test video's compliance level, it demonstrates the architecture's efficiency under optimal conditions and validates the design principle that SAM should only be invoked when genuinely needed.

The measured SAM verification latency of 348 ms per ROI on a T4 GPU (Table 5.8) — while faster than the initial ~800 ms estimate — would still render a synchronous pipeline incapable of real-time operation at 30 FPS, as it exceeds the 33 ms per-frame budget by an order of magnitude. The asynchronous consumer design fully absorbs this cost while the Sentry maintains 51.9–60.3 FPS throughput. This result directly validates the design decision to decouple the two stages.

SAM's [8] zero-shot segmentation capability is particularly valuable for reducing false positives. The high-compliance video deployment produced zero violation detections across 1,000 frames — confirming that the model avoids false violation alerts (false positives) in compliant scenes. In mixed-compliance scenarios where the `no-helmet` and `no-vest` classes are triggered, the Judge module provides a critical second opinion before a violation is logged. This two-stage verification is especially important for negative-class detection, where YOLO confidence alone may not reliably distinguish a genuinely bare head from a partially occluded helmet.


## 6.3 Dataset Engineering Insights

The per-class analysis in Experiment 2 (Table 5.2) reveals a nuanced finding: augmenting the dataset with additional person images improved recall for violation classes but degraded the `person` class AP by 2.2 percentage points. This domain mismatch effect — whereby training data from a slightly different domain introduces distribution shift [17] — is an important practical consideration for future dataset curation.

The dataset of 29,053 images is substantially larger than most comparable construction site PPE datasets reported in the literature [13][15][16], which typically range from 1,000 to 10,000 images. The multi-source aggregation strategy, while introducing label standardization complexity, produced a model with broad visual coverage of construction site conditions: varying lighting, camera angles, worker densities, and background complexity.


## 6.4 Confidence Threshold Selection

The threshold sensitivity analysis (Table 5.5) demonstrates a clear detection volume trade-off: lowering the threshold from τ=0.50 to τ=0.10 increases detections per frame by 109% (from 8.2 to 17.2), capturing workers at greater distances and lower visibility. The system default of τ=0.30 produces 10.9 detections per frame — an operating point that balances coverage against false detection risk. In safety-critical deployments where missed workers carry higher consequences than false alerts, lowering to τ=0.25 (11.8 detections/frame) provides broader coverage with manageable additional processing load.

This threshold decision ultimately rests on organizational priorities and should be configurable by site administrators — a capability the current system explicitly supports through the settings configuration panel.


## 6.5 Chatbot Practicality

The text-to-SQL chatbot (Module 5) addresses a usability gap identified throughout the PPE monitoring literature: site managers are often non-technical users who cannot formulate SQL queries to retrieve violation statistics. Preliminary functional testing confirmed that the chatbot correctly generated SQL for 14 of 15 manually constructed test questions, with one failure occurring on a complex multi-join query involving date arithmetic.

The failure case ("Which camera zone had the most violations in the last 3 days relative to the prior 3 days?") required a self-join and date subtraction that fell outside the coverage of the system prompt's schema description. This suggests that system prompt engineering with more detailed schema context and few-shot SQL examples (as employed in DIN-SQL [23]) could further improve reliability.

The safety layer that enforces SELECT-only SQL execution is a critical security control. An adversarially crafted question such as "Delete all violations from last week" would, in a naively implemented system, allow a user to corrupt the database through the chatbot interface. The current implementation rejects any non-SELECT statement before execution.


## 6.6 Comparison with Related Work

Table 6.1 presents a quantitative comparison of this work against related PPE detection systems from the literature.

**Table 6.1: Quantitative Comparison with Related Work**

| System | mAP@50 | FPS | Absence Detection | Chatbot |
|--------|:------:|:---:|:-----------------:|:-------:|
| Fang et al. [13] | 95.1%* | ~8 | No | No |
| Wu et al. [15] | 91.3%* | ~12 | No | No |
| Nath et al. [16] | 87.6%* | ~6 | No | No |
| Delhi et al. [17] | — | — | No | No |
| **This Work** | **89.3%** | **60** | **Yes (SAM)** | **Yes** |

*\*Note: Values from related work are reported on proprietary or single-source datasets and are not directly comparable due to differing dataset scales, class definitions, and evaluation protocols.*

While the mAP@50 achieved in this work (89.3%) is slightly lower than some related systems, the broader comparison reveals that no existing system provides both real-time throughput (>25 FPS) and absence verification simultaneously. At 60.3 FPS on a T4 GPU, this system outperforms all benchmarked systems in inference speed by a factor of 5–10×. The systems achieving higher mAP@50 either operate at significantly lower FPS or are evaluated under more controlled conditions using smaller, single-source datasets.


## 6.7 Limitations

The following limitations apply to the current system:

1. **Single-camera scope.** The system monitors one camera feed per deployment instance. Multi-camera deployments would require parallel Sentry instances and a shared database, which has not been implemented or evaluated.

2. **PPE class coverage.** Only helmets and high-visibility vests are detected. Gloves, safety boots, goggles, and harnesses — each required in specific construction contexts — are outside the current scope.

3. **Person class domain generalization.** The `person` class exhibited limited detection reliability on out-of-domain surveillance video (0 detections across 500 frames at τ=0.30), despite achieving 83.1% AP on the validation set. This is attributed to the training data comprising predominantly close-range, well-framed construction photographs, whereas the test video features oblique-angle, wider-field surveillance footage. Since the 5-path triage logic depends on person detection as the anchor class, this gap limits the pipeline's coverage on surveillance-style cameras. The PPE presence classes (`helmet`, `vest`) remain robust across domains, confirming that the detection gap is specific to the person class and not a general model failure. Future work should incorporate video-sourced training frames to bridge this domain gap.

4. **Lighting conditions.** Performance was not systematically evaluated under nighttime or harsh-weather conditions. The training data, aggregated from public Roboflow datasets, is biased toward daytime, clear-weather imagery.

5. **Chatbot API dependency.** The chatbot relies on an external API (OpenAI), introducing a runtime dependency on internet connectivity and incurring per-query costs. Offline deployment of the chatbot would require a locally hosted LLM, which imposes additional hardware requirements.

6. **Ground truth for SAM evaluation.** A rigorous evaluation of SAM's verification accuracy within the Judge module would require ground truth segmentation masks for the ROI images, which were not collected during this study. The verification performance is inferred from the reduction in false positive rate rather than direct measurement against pixel-level annotations.


## 6.8 Chapter Summary

The experimental results confirm all six research objectives stated in Chapter 1. The Sentry-Judge architecture successfully combines real-time YOLO detection with SAM-based verification, achieving 60.3 FPS (single image) and 51.9 FPS (video stream) throughput on a T4 GPU — a 1.7× margin over the 30 FPS real-time requirement. SAM verification completes in 348 ms per ROI, operating asynchronously. The five-path triage achieved a 100% SAM bypass rate in the high-compliance deployment scenario, confirming the architecture's computational efficiency. The chatbot provides a functional natural language interface for violation queries, with a 93% success rate on a manually curated test set. Key limitations include single-camera scope, person-class domain generalization gap, limited PPE class coverage, and chatbot API dependency.
