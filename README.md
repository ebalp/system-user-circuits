# System-User Circuits

Mechanistic interpretability study of how LLMs resolve conflicts between system prompts and user messages. The goal is to isolate the internal circuits that mediate instruction hierarchy — when system and user instructions conflict, what determines which one the model follows?

This work is motivated by AI safety: understanding these circuits can reveal vulnerabilities to jailbreaking, prompt injection, and adversarial steering, and inform the design of robust defenses.

## Core Hypotheses

- **H1 (Circuitry):** Causal circuits govern system-user conflict resolution. Patching these circuits flips the winner of the conflict.
- **H2 (Refusal Overlap):** There is measurable overlap between conflict resolution circuits and refusal circuits.
- **H3 (Directionality):** A single "precedence direction" mediates conflict resolution. Projecting activations onto it predicts the outcome; steering along it flips it.

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **0 — Behavioral Analysis** | Construct conflict dataset, evaluate System Compliance Rate (SCR) across models and conditions, select targets for mechanistic analysis. | In progress |
| **1 — Representational Analysis** | Linear probing, logit attribution, and attention analysis to determine whether and where the model represents the conflict resolution decision. | Planned |
| **2 — Causal Intervention** | Activation patching and steering to test H1 and H3 — can we causally flip which instruction wins? | Planned |
| **3 — Refusal Overlap & Robustness** | Test H2 (overlap with refusal circuits) and validate findings through systematic ablations. | Planned |

**Stretch goal:** Extend to multiturn conversations where the user progressively attempts to override system instructions across turns.

Each phase has its own directory with setup instructions — see the README in each `phase*/` folder.
