# Ethics and Data Usage Statement

## Data Sources

All datasets used in the training, validation, and testing of the YOLO26m model in this work were obtained from Roboflow Universe, a publicly accessible platform for computer vision datasets. The seven datasets employed are:

1. Construction PPE Detection [Roboflow, huiyao-hu-sj18e]
2. Construction PPE Detection OIYSP [Roboflow, construction-plxig]
3. Construction Worker PPE [Roboflow, human-posture]
4. Construction PPE QOFI4 [Roboflow, gaos-workspace]
5. Safety AKUCZ [Roboflow, construction-propn]
6. Construction RINEU [Roboflow, envisage]
7. Construction Person Detection XC5FZ [Roboflow, person-detection-pmyzi]

Each dataset is published under an open or Creative Commons license permitting academic use. No personal data, biometric data, or personally identifiable information (PII) was collected or used in this research.

## Privacy Considerations

The detection system, when deployed, captures and processes video frames from construction site cameras. The following privacy safeguards are incorporated into the system design:

- **No biometric data** is collected or stored. The system records bounding box coordinates and class labels only; no facial recognition or worker identification is performed.
- **Violation ROI images** (cropped head and torso regions) are stored locally on the edge device and are not transmitted to external servers, except when the OpenAI API is invoked for chatbot queries (which receive only text queries, not images).
- **Video frames** are processed in memory and are not persistently stored by the monitoring pipeline.

## Human Subjects

This research does not involve human subjects studies, surveys, interviews, or any form of direct interaction with human participants. All evaluation was performed on pre-existing annotated image datasets. Accordingly, no institutional ethics board approval was required for this work.

## Responsible AI Statement

The violation chatbot module uses the OpenAI GPT-4o-mini API to translate natural language questions into SQL queries. The system enforces a strict read-only database policy — the chatbot cannot modify, delete, or insert records. All generated queries are validated before execution. The system is designed to assist, not replace, human safety officers, and all violation reports require human review before any disciplinary action is taken.
