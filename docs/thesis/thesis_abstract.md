# Abstract

**Intelligent PPE Compliance Monitoring System for Construction Site Safety Using Deep Learning and Natural Language Processing**

Construction sites represent one of the highest-risk occupational environments globally, with the sector recording 1,075 fatalities in 2023 in the United States alone. Personal Protective Equipment (PPE) — specifically hard hats and high-visibility vests — constitutes a primary line of defence against workplace injury, yet continuous compliance monitoring remains a practical challenge for site managers relying on manual inspection.

This thesis presents an Intelligent PPE Compliance Monitoring System that addresses three key limitations of existing automated approaches: the inability to reliably detect the *absence* of protective equipment (the absence detection paradox), excessive alert generation leading to safety officer disengagement, and the computational impracticality of deploying dense verification models in real-time video pipelines.

The proposed system employs a decoupled "Sentry-Judge" microservice architecture. The Sentry module — built on the YOLO26m object detection model trained on a unified dataset of 29,053 construction site images across five classes — applies a novel five-path intelligent triage algorithm to classify each detected person. Only uncertain detections (those where helmet or vest presence cannot be determined with confidence) are forwarded to the Judge module, which employs the Segment Anything Model (SAM) for pixel-level verification in an asynchronous background thread, preserving real-time throughput.

Experimental evaluation demonstrates that the YOLO26m model achieves 89.3% mAP@50 and 92.3% Average Precision for the safety-critical no-helmet class, trained over 50 epochs. The system further incorporates an LLM-based daily report generator producing OSHA-format PDF compliance summaries, and a text-to-SQL violation chatbot that enables site managers to query violation records in natural language without SQL knowledge.

The system demonstrates that real-time PPE absence verification is achievable within the computational budget of a single-GPU edge device, offering a practical and deployable solution for automated construction site safety compliance.

**Keywords:** PPE detection, construction site safety, object detection, Segment Anything Model, YOLO, text-to-SQL, deep learning, computer vision
