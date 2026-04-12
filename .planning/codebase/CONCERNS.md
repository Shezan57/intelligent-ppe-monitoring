# Concerns and Technical Debt

## Bugs & Known Issues
- Currently, there are ongoing debugging threads related to the asynchronous message queue between the YOLO (Sentry) and SAM (Judge) nodes where the stream execution might be blocked or SAM verification metrics are missed.

## Technical Debt
- **Testing Coverage**: As verified, local backend directory `/tests` is mostly empty, pointing towards a lack of native CI-ready integration or unit tests. System heavily relies on manual colab notebooks and isolated test scripts.
- **Complexity**: Multiple model management via hybrid architectures causes complexities in deduplication and tracking if confidence thresholds shift.
