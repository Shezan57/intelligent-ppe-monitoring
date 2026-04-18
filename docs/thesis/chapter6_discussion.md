# Chapter 6: Discussion

## 6.1 Introduction

This chapter interprets the experimental findings presented in Chapter 5, discusses  the system's contributions relative to existing work, examines its limitations, and considers broader implications for automated construction site safety monitoring.


## 6.2 Analysis of the Sentry-Judge Architecture

The core architectural contribution of this work is the decoupled Sentry-Judge pipeline, and the experimental results substantiate its design rationale. The Sentry's five-path triage mechanism proved highly effective in practice: 83.6% of all person detections in the test video were resolved via fast paths (Paths 1 and 2) without SAM invocation (Table 5.4). This bypass rate is higher than the 80% estimate established during system design, indicating that the training data distribution is well-calibrated for real-world construction site conditions.

The SAM verification latency of ~800 ms per ROI, which would render a synchronous pipeline incapable of real-time operation, is fully absorbed by the asynchronous consumer while the Sentry maintains 47 FPS throughput. This result directly validates the design decision to decouple the two stages. Systems that integrate SAM synchronously into the detection loop — a conceptually simpler approach — could not achieve the same throughput without a cluster of GPU servers.

SAM's [8] zero-shot segmentation capability proved particularly valuable for reducing false positives. In preliminary testing with the Sentry alone (without Judge verification), the false positive rate for the `no-vest` class reached 23%. After Judge verification, confirmed false positives fell below 5%, a reduction of over 78%. This finding underscores the practical importance of two-stage verification for negative-class detection, and supports the use of foundation models for domain-specific verification without task-specific fine-tuning [8].


## 6.3 Dataset Engineering Insights

The per-class analysis in Experiment 2 (Table 5.2) reveals a nuanced finding: augmenting the dataset with additional person images improved recall for violation classes but degraded the `person` class AP by 2.2 percentage points. This domain mismatch effect — whereby training data from a slightly different domain introduces distribution shift [17] — is an important practical consideration for future dataset curation.

The dataset of 29,053 images is substantially larger than most comparable construction site PPE datasets reported in the literature [13][15][16], which typically range from 1,000 to 10,000 images. The multi-source aggregation strategy, while introducing label standardization complexity, produced a model with broad visual coverage of construction site conditions: varying lighting, camera angles, worker densities, and background complexity.


## 6.4 Confidence Threshold Selection

The threshold sensitivity results (Table 5.3) highlight a fundamental trade-off in safety monitoring: choosing threshold 0.30 over 0.25 increases precision by 1.6 percentage points but reduces recall by 3.4 percentage points. In a safety context, false negatives (missed violations) carry a higher human cost than false positives (spurious alerts). On this basis, the 0.25 threshold may be more appropriate for deployment environments where safety compliance is the primary concern, while the 0.30 threshold is recommended where operator trust and alert reliability are more critical.

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
| **This Work** | **89.3%** | **47** | **Yes (SAM)** | **Yes** |

*\*Note: Values from related work are reported on proprietary or single-source datasets and are not directly comparable due to differing dataset scales, class definitions, and evaluation protocols.*

While the mAP@50 achieved in this work (89.3%) is slightly lower than some related systems, the broader comparison reveals that no existing system provides both real-time throughput (>25 FPS) and absence verification simultaneously. The systems achieving higher mAP@50 either operate at significantly lower FPS or are evaluated under more controlled conditions using smaller, single-source datasets.


## 6.7 Limitations

The following limitations apply to the current system:

1. **Single-camera scope.** The system monitors one camera feed per deployment instance. Multi-camera deployments would require parallel Sentry instances and a shared database, which has not been implemented or evaluated.

2. **PPE class coverage.** Only helmets and high-visibility vests are detected. Gloves, safety boots, goggles, and harnesses — each required in specific construction contexts — are outside the current scope.

3. **Lighting conditions.** Performance was not systematically evaluated under nighttime or harsh-weather conditions. The training data, aggregated from public Roboflow datasets, is biased toward daytime, clear-weather imagery.

4. **Chatbot API dependency.** The chatbot relies on an external API (OpenAI), introducing a runtime dependency on internet connectivity and incurring per-query costs. Offline deployment of the chatbot would require a locally hosted LLM, which imposes additional hardware requirements.

5. **Ground truth for SAM evaluation.** A rigorous evaluation of SAM's verification accuracy within the Judge module would require ground truth segmentation masks for the ROI images, which were not collected during this study. The verification performance is inferred from the reduction in false positive rate rather than direct measurement against pixel-level annotations.


## 6.8 Chapter Summary

The experimental results confirm all six research objectives stated in Chapter 1. The Sentry-Judge architecture successfully combines real-time YOLO detection with SAM-based verification, achieving 47 FPS throughput while reducing false positives by 78%. The five-path triage routes 83.6% of detections through fast paths, minimizing computational overhead. The chatbot provides a functional natural language interface for violation queries, with a 93% success rate on a manually curated test set. Key limitations include single-camera scope, limited PPE class coverage, and chatbot API dependency.
