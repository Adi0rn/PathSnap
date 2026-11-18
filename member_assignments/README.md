# Member A: data and features

This branch adds the assigned baseline source and tests from the uploaded scaffold. The implementation tasks and known issues remain in [MEMBER_A.md](MEMBER_A.md). See [four-person integration guide](../docs/FOUR_PERSON_INTEGRATION.md) for ownership and dependencies.

The [original handoff ZIP](../handoffs/PathSnap_Member_A_Data_and_Features.zip) is also included. Its owned/ files are installed at their project paths on this branch; archive-specific references in the brief refer to that ZIP. This partial contribution needs the full shared scaffold and local raw dataset for execution. No datasets or model weights are added here.

Verification: 11 existing data/features tests passed against the full uploaded scaffold on 18 November 2026 using `python -m pytest tests/test_data tests/test_features -q -p no:cacheprovider`. This validates the assigned baseline tests; it does not mean the listed improvement tasks are complete.
