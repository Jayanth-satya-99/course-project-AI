# Policy-Adherent Agent: Master Project Reference & Architecture Guide

This document encapsulates the complete architecture, module interactions, data flows, and design patterns for the **Policy-Adherent Agent** project. Use this as a self-contained reference guide when building, extending, or debugging your pipeline.

---

## 1. High-Level Project Overview & Goal
The goal of this project is to implement a modular AI pipeline that handles customer support queries while strictly enforcing organizational policy constraints and actively resisting adversarial or manipulative user behavior.

### The 5 Core Modules
1. **Module 1 (Search, CSP, and Minimax):** The logical rule engine and tactical defender. Uses graph search and Constraint Satisfaction Problems (CSP) to evaluate rules, and Minimax search with Alpha-Beta Pruning to manage live conversational defense against adversarial users.
2. **Module 2 (Sequential Planning):** The safety sequencer. Models claim resolution as a planning problem, ensuring irreversible actions (like issuing a refund) only occur after all preconditions and verification steps are met.
3. **Module 3 (Reasoning under Uncertainty):** The risk assessor. Uses probabilistic graphical models (e.g., Bayesian networks) to estimate fraud probability and evaluate evidence reliability (missing, weak, contradictory, or fabricated).
4. **Module 4 (Reinforcement Learning & NLU):** The grand strategist and parser. The current prototype uses regex/heuristic NLU and a discrete Q-table with safe heuristic priors and optional online updates to select overarching actions (approve, reject, investigate, escalate).
5. **Module 5 (Response Generation & Multi-Agent Environment):** Ollama generates a short empathy-only opening by default; the decision, policy citations, evidence status, and next steps remain policy-grounded templates. The module also supports multi-agent red-teaming simulations.

---

## 2. Core Architectural Principles & Clarifications

* **Probabilities from Module 3 vs. MDP Transitions:** 
  Module 3's probabilities (e.g., $P(\text{Fraud} \mid \text{Evidence})$) are **not** transition probabilities (the physical chance of the environment moving from state A to state B). Instead, they are **features within the State space ($S$)**. They act as a mathematical "belief state" representing uncertainty about the customer, turning a Partially Observable environment into a Belief-MDP.
* **Module 4 vs. Module 1 (Minimax) Hierarchy:** 
  They do not compete. **Module 4 is the Strategist** that decides the macro-level goal for the turn (e.g., *Ask for additional evidence*). **Module 1's Minimax is the Tactical Defender** that takes that goal and figures out the safest conversational phrasing to execute it without succumbing to the user's emotional pressure or threats.
* **CSP Uncertainty Trigger:** 
  When Module 1's CSP flags a variable as `UNCERTAIN` or `POTENTIALLY_MANIPULATED`, it does not guess the value; it triggers Module 3 to quantify the risk, which in turn informs Module 4's strategic decision on how to investigate.

### Current Prototype Status and Limits

* The Bayesian network performs exact enumeration using authored CPTs; those probabilities are illustrative and have not been calibrated on real claims.
* The Q-table has safe heuristic priors and can update online, but it is not a pretrained or dataset-trained RL model. Evaluation runs with learning disabled.
* NLU is regex/heuristic-based. The default local Ollama model (`llama3.2:3b`) writes only an empathy acknowledgment; deterministic code retains control over the decision and policy details.
* Transaction and customer-history lookups use a small in-memory demo database. The web verification controls record simulated reviewer decisions; they are not authenticated and do not connect to real order, evidence, or staff systems.
* Customer statements that they possess a receipt or photo are not treated as verified evidence. Refund approval is blocked until required checks are recorded, and manager approval is only accepted after order, evidence, and history checks for high-value refunds.
* The benchmark contains a small synthetic scenario set. Its metrics are not evidence of production accuracy or real-world fraud detection performance.

### Running the Local Response Model

Install Ollama separately, then pull the default model and start its local server:

```powershell
ollama pull llama3.2:3b
ollama serve
```

The app connects to `http://127.0.0.1:11434` by default. Configure `OLLAMA_MODEL` or `OLLAMA_BASE_URL` to use another local Ollama model or endpoint. If Ollama is unavailable, the chat API returns an explicit `503` error; it does not silently substitute template text. Set `GENERATOR_PROVIDER=template` to run fully deterministic offline tests or the benchmark, which intentionally isolates policy decisions from text generation.

---

## 3. Step-by-Step Data Flow Pipeline

1. **Ingestion & Extraction (Module 4/5 NLU):** Raw customer text is parsed into a structured JSON payload containing claims, order amounts, dates, evidence references, and adversarial tactics (e.g., threats of chargebacks, emotional appeals).
2. **Rule Mapping & CSP Evaluation (Module 1):** Extracted data is searched against policy graphs. The CSP solver assigns status labels to every required constraint: `SATISFIED`, `VIOLATED`, `UNCERTAIN`, or `POTENTIALLY_MANIPULATED`.
3. **Risk & Uncertainty Assessment (Module 3):** If any constraint is `UNCERTAIN` or `POTENTIALLY_MANIPULATED`, Module 3 computes posterior fraud probabilities and evidence reliability metrics.
4. **State Construction & Strategic Decision (Module 4):** CSP constraint statuses, Bayesian fraud probabilities, and investigation costs form the RL agent's state vector $S_t$. The agent selects the optimal macro action (e.g., `REQUEST_EVIDENCE`).
5. **Sequential Precondition Check (Module 2):** The planner verifies whether preconditions for the chosen action are met, blocking premature irreversible actions and establishing a verification plan.
6. **Tactical Conversational Defense (Module 1 Minimax):** Given Module 4's goal, Module 1 builds a depth-limited dialogue tree, using Alpha-Beta pruning to discard non-compliant responses and select the safest tactical reply.
7. **Grounded Response Generation (Module 5 GenAI):** The winning tactical move is rendered into empathetic, natural language explicitly citing policy clauses and evidence gaps.

---

## 4. Architectural Implementation Patterns

### Pattern 1: Synchronous Sequential Pipeline (Strict Waterfall)
* **How it works:** Every conversational turn executes all modules sequentially in a fixed assembly line (NLU $\rightarrow$ M1 CSP $\rightarrow$ M3 Bayes $\rightarrow$ M4 RL $\rightarrow$ M2 Planner $\rightarrow$ M1 Minimax $\rightarrow$ M5 GenAI).
* **Pros:** Easiest to implement, debug, and audit; guarantees 100% deterministic traceability for compliance.
* **Cons:** Head-of-line blocking; re-runs heavy probabilistic calculations even on routine conversational turns.

### Pattern 2: RL-Centric Orchestrator Hub (Tool-Calling)
* **How it works:** Module 4 acts as an autonomous agent whose action space includes *internal epistemic actions* (e.g., `CALL_CSP`, `CALL_BAYES`) and *external communicative actions* (e.g., `ASK_EVIDENCE`, `APPROVE`). It queries modules as tools only when needed.
* **Pros:** Dynamically adapts computation to claim complexity; saves cycles on straightforward requests.
* **Cons:** Requires careful reward shaping and action masking to prevent infinite internal tool loops.

### Pattern 3: Event-Driven Asynchronous Blackboard
* **How it works:** Modules attach as async readers/writers to a central shared memory document (the Claim Blackboard) coordinated via pub/sub events and epoch version barriers.
* **Pros:** Highly scalable; ideal for multi-agent red-team simulations and parallel background forensics.
* **Cons:** High engineering complexity; prone to race conditions and stale reads if epoch locks are missing.

### Pattern 4: Guardrailed Dual-Loop (Fast System 1 / Slow System 2) — *Recommended Hybrid*
* **How it works:** 
  * **Loop 1 (Fast Tactical):** Stays inside Module 1's Minimax tree and Module 5 to handle multi-turn adversarial arguments while keeping the strategic goal locked.
  * **Loop 2 (Slow Strategic):** Wakes up Modules 3 and 4 *only* when factual state transitions occur (e.g., new evidence uploaded, verification passed, or Minimax stalemate reached).
* **Pros:** Combines sub-second conversational responsiveness with rigorous multi-module policy enforcement.

---

## 5. Recommended Build Order
1. **Module 1 (Search, CSP, and Minimax):** *(Already Built)* Establish your policy graphs and constraint satisfaction solver.
2. **Module 3 (Reasoning under Uncertainty):** Build your Bayesian network to consume `UNCERTAIN` CSP flags and output fraud probabilities.
3. **Module 4 (Reinforcement Learning):** Formulate the state vector from Modules 1 & 3 and implement your Q-learning or Contextual Bandit action selector.
4. **Module 2 (Sequential Planning):** Wrap action execution with PDDL-style precondition checks.
5. **Module 5 (Generative AI & Interface):** Wire up the conversational wrapper and grounded explanation generator.