# RL-Centric Orchestrator Hub (Pattern 2 Implementation)
# ========================================================
# Coordinates the 5 AI modules following the RL-Centric architecture:
#
#   User Input
#       │
#       ▼
#   [Module 4 NLU] ──▶ Structured Claim
#       │
#       ▼
#   [Module 4 RL Strategist Loop]
#       ├──▶ [Tool: Module 1 CSP] (Evaluates policy graph constraints)
#       ├──▶ [Tool: Module 3 Bayes] (Computes P(Fraud | Evidence))
#       └──▶ Selects External Macro Action (e.g. ASK_FOR_EVIDENCE, APPROVE, REJECT)
#               │
#               ▼
#       [Module 2 Planner] (Safety Sequencer: Blocks premature irreversible actions)
#               │
#               ▼
#       [Module 1 Minimax] (Tactical Defender: Alpha-Beta pruning for safe dialogue)
#               │
#               ▼
#       [Module 5 GenAI] (The Voice: Grounded, empathetic customer explanation)

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
from dataclasses import replace

from shared.models import (
    ActionType, ClaimExtraction, CSPResult, RiskAssessment,
    PlanResult, MinimaxResult, GeneratedResponse, ConversationState,
    ConstraintStatus, ThreatLevel
)
from module1.policy_graph import PolicyGraph
from module1.csp import CSPSolver
from module1.minimax import MinimaxEngine
from module2.planner import SequentialPlanner
from module3.bayesian import BayesianRiskAssessor
from module4.nlu import NLUExtractor
from module4.rl_agent import RLAgent
from module5.generator import ResponseGenerator


class PolicyAdherentOrchestrator:
    """
    Master orchestrator implementing Pattern 2 (RL-Centric Tool-Calling Hub).
    Maintains active conversation sessions and runs the full 5-module pipeline.
    """

    def __init__(self, generator: Optional[ResponseGenerator] = None):
        # Initialize the 5 core modules
        self.policy_graph = PolicyGraph()
        self.csp_solver = CSPSolver(self.policy_graph)
        self.evidence_graph = self.csp_solver.evidence_graph
        self.minimax_engine = MinimaxEngine()
        self.planner = SequentialPlanner()
        self.bayes_assessor = BayesianRiskAssessor()
        self.nlu = NLUExtractor()
        self.rl_agent = RLAgent()
        self.generator = generator or ResponseGenerator()

        # Session storage: session_id -> ConversationState
        self.sessions: Dict[str, ConversationState] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationState:
        """Retrieves an existing conversation session or initializes a fresh one."""
        s_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        if s_id not in self.sessions:
            self.sessions[s_id] = ConversationState(session_id=s_id)
        return self.sessions[s_id]

    def reset_session(self, session_id: str):
        """Resets a conversation session to start clean."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def complete_verification(self, session_id: str, verification: str) -> ConversationState:
        """Record a completed administrative verification for a live claim."""
        conv = self.sessions.get(session_id)
        if conv is None or conv.claim is None:
            raise ValueError(f"No active claim exists for session '{session_id}'.")
        if verification not in {
            "order_lookup", "evidence_check", "history_check",
            "manager_approval", "item_returned",
        }:
            raise ValueError(f"Unsupported verification step: {verification}")
        if verification == "order_lookup":
            if not conv.claim.order_id or not self.csp_solver.get_order_record(conv.claim.order_id):
                raise ValueError("The order ID could not be verified in the transaction records.")
        if verification == "evidence_check" and not conv.claim.evidence_provided:
            raise ValueError("Evidence cannot be verified until the customer provides it.")
        if verification == "manager_approval" and (
            conv.claim.amount <= 200 or conv.claim.intent == "cancellation"
        ):
            raise ValueError("Manager approval is only required for refunds above $200.")
        if verification == "manager_approval":
            prerequisites = {"order_lookup", "evidence_check", "history_check"}
            missing = sorted(prerequisites - set(conv.completed_verifications))
            if missing:
                raise ValueError(
                    "Manager approval requires completed verification: "
                    + ", ".join(missing)
                    + "."
                )
        if verification not in conv.completed_verifications:
            conv.completed_verifications.append(verification)
        return conv

    @staticmethod
    def _merge_claims(previous: ClaimExtraction,
                      incoming: ClaimExtraction) -> ClaimExtraction:
        """Carry verified claim context across turns and flag changed core facts."""
        tactics = list(dict.fromkeys(
            previous.adversarial_tactics + incoming.adversarial_tactics
        ))

        contradictory_claims = list(previous.contradictory_claims)
        for c in incoming.contradictory_claims:
            if c not in contradictory_claims:
                contradictory_claims.append(c)

        # Cross-turn contradiction checks between previous and incoming claims
        if (
            previous.amount > 0 and incoming.amount > 0 and
            abs(previous.amount - incoming.amount) > 0.01
        ):
            mismatch = {
                "type": "amount_mismatch",
                "source": "cross_turn",
                "claim_a": f"${previous.amount:.2f}",
                "claim_b": f"${incoming.amount:.2f}",
                "description": f"Claimed amount changed from ${previous.amount:.2f} to ${incoming.amount:.2f} across conversation turns.",
            }
            if mismatch not in contradictory_claims:
                contradictory_claims.append(mismatch)

        if previous.order_id and incoming.order_id and previous.order_id != incoming.order_id:
            mismatch = {
                "type": "order_mismatch",
                "source": "cross_turn",
                "claim_a": previous.order_id,
                "claim_b": incoming.order_id,
                "description": f"Order ID changed from {previous.order_id} to {incoming.order_id} across conversation turns.",
            }
            if mismatch not in contradictory_claims:
                contradictory_claims.append(mismatch)

        if previous.purchase_date and incoming.purchase_date and previous.purchase_date != incoming.purchase_date:
            mismatch = {
                "type": "date_mismatch",
                "source": "cross_turn",
                "claim_a": previous.purchase_date,
                "claim_b": incoming.purchase_date,
                "description": f"Purchase date changed from {previous.purchase_date} to {incoming.purchase_date} across conversation turns.",
            }
            if mismatch not in contradictory_claims:
                contradictory_claims.append(mismatch)

        has_contradictions = bool(contradictory_claims)
        if has_contradictions and "contradiction" not in tactics:
            tactics.append("contradiction")

        # Merge supporting evidence
        merged_supporting = list(previous.supporting_evidence)
        seen_evidence_types = {e["evidence_type"] for e in merged_supporting if "evidence_type" in e}
        for item in incoming.supporting_evidence:
            if item.get("evidence_type") not in seen_evidence_types:
                merged_supporting.append(item)
                seen_evidence_types.add(item.get("evidence_type"))

        # Merge emotional appeals and threats
        emotional_appeals = list(dict.fromkeys(previous.emotional_appeals + incoming.emotional_appeals))
        threats = list(dict.fromkeys(previous.threats + incoming.threats))
        persuasive_strategies = list(dict.fromkeys(previous.persuasive_strategies + incoming.persuasive_strategies))
        if has_contradictions and "fact_shifting" not in persuasive_strategies:
            persuasive_strategies.append("fact_shifting")

        threat_order = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
        }
        return replace(
            incoming,
            intent=incoming.intent if incoming.intent != "complaint" else previous.intent,
            amount=(
                previous.amount if has_contradictions and previous.amount > 0
                else incoming.amount or previous.amount
            ),
            item=incoming.item if incoming.item != "item" else previous.item,
            purchase_date=incoming.purchase_date or previous.purchase_date,
            order_id=(
                previous.order_id if has_contradictions and previous.order_id
                else incoming.order_id or previous.order_id
            ),
            evidence_provided=list(dict.fromkeys(
                previous.evidence_provided + incoming.evidence_provided
            )),
            threat_level=max(
                (previous.threat_level, incoming.threat_level),
                key=threat_order.get,
            ),
            adversarial_tactics=tactics,
            raw_text=f"{previous.raw_text}\n{incoming.raw_text}",
            claimed_previous_claims=(
                incoming.claimed_previous_claims
                if incoming.claimed_previous_claims is not None
                else previous.claimed_previous_claims
            ),
            contradictory_claims=contradictory_claims,
            supporting_evidence=merged_supporting,
            emotional_appeals=emotional_appeals,
            threats=threats,
            persuasive_strategies=persuasive_strategies,
            conversation_history=incoming.conversation_history or previous.conversation_history,
        )

    def process_claim_file(self, claim_file: Union[str, Path],
                           output_dir: Optional[Union[str, Path]] = None,
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full pipeline taking input from a ClaimExtraction JSON file,
        saving intermediate artifacts (csp_result.json, risk_assessment.json)
        and final response (pipeline_response.json) into output_dir.
        """
        claim_path = Path(claim_file)
        if not claim_path.exists():
            raise FileNotFoundError(f"Claim file not found: {claim_file}")

        claim = ClaimExtraction.from_file(claim_path)
        out_dir = Path(output_dir) if output_dir else claim_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        conv = self.get_or_create_session(session_id)
        conv.turn += 1
        conv.claim = claim
        if claim.order_id and self.csp_solver.get_order_record(claim.order_id):
            if "order_lookup" not in conv.completed_verifications:
                conv.completed_verifications.append("order_lookup")

        # Step 1: CSP Evaluation (Module 1)
        evidence_graph_res = self.evidence_graph.search(
            claim, conv.completed_verifications,
            self.csp_solver.get_order_record(claim.order_id) if claim.order_id else None
        )
        csp_result = self.csp_solver.evaluate(claim, conv.completed_verifications, evidence_graph_res)
        conv.csp_result = csp_result
        csp_out_path = out_dir / "csp_result.json"
        csp_result.to_file(csp_out_path)

        # Step 2: Bayesian Risk Assessment (Module 3)
        risk = self.bayes_assessor.assess_risk(claim, csp_result, conv.completed_verifications)
        conv.risk_assessment = risk
        risk_out_path = out_dir / "risk_assessment.json"
        risk.to_file(risk_out_path)

        # Step 3: Strategic RL Action Selection (Module 4) & Sequential Planning (Module 2)
        action, q_vals = self.rl_agent.select_action(
            conv, csp_queried=True, bayes_queried=True, force_exploitation=True
        )
        conv.current_action = action
        plan_result = self.planner.generate_plan("case_resolved", conv)
        conv.plan_result = plan_result

        # Step 4: Minimax Tactical Search (Module 1)
        applicable_clauses = csp_result.applicable_policies
        minimax_result = self.minimax_engine.search(action, claim, applicable_clauses)
        conv.minimax_result = minimax_result

        # Step 5: Grounded Explanation Generation (Module 5)
        try:
            response = self.generator.generate(
                action=action,
                claim=claim,
                csp_result=csp_result,
                risk_assessment=risk,
                plan_result=plan_result,
                minimax_result=minimax_result,
            )
            response_text = response.text
            cited_policies = response.cited_policies
            evidence_status = response.evidence_status
        except Exception:
            from module5.generator import ResponseGenerator
            fallback_gen = ResponseGenerator(provider="template")
            response = fallback_gen.generate(
                action=action,
                claim=claim,
                csp_result=csp_result,
                risk_assessment=risk,
                plan_result=plan_result,
                minimax_result=minimax_result,
            )
            response_text = response.text
            cited_policies = response.cited_policies
            evidence_status = response.evidence_status

        trace_out = {
            "session_id": conv.session_id,
            "claim_file": str(claim_path),
            "csp_result_file": str(csp_out_path),
            "risk_assessment_file": str(risk_out_path),
            "action": action.value if hasattr(action, "value") else str(action),
            "plan_approved": plan_result.action_approved,
            "response_text": response_text,
            "cited_policies": cited_policies,
            "evidence_status": evidence_status,
        }
        response_out_path = out_dir / "pipeline_response.json"
        response_out_path.write_text(json.dumps(trace_out, indent=2), encoding="utf-8")
        trace_out["pipeline_response_file"] = str(response_out_path)
        return trace_out

    def process_message(self, user_text: str, session_id: Optional[str] = None,
                        learn: bool = True) -> Dict[str, Any]:
        """
        Executes one full turn of the Policy-Adherent Agent pipeline.
        Returns a rich execution trace containing the internal thoughts,
        intermediate module states, and final customer response.
        """
        conv = self.get_or_create_session(session_id)
        conv.turn += 1

        # Execution trace for real-time visualization and inspection
        trace = {
            "session_id": conv.session_id,
            "turn": conv.turn,
            "user_input": user_text,
            "pipeline_steps": [],
            "internal_tool_calls": []
        }

        # -----------------------------------------------------------------------
        # Step 1: NLU Ingestion & Extraction (Module 4)
        # -----------------------------------------------------------------------
        claim = self.nlu.extract(user_text, conv.conversation_history)
        if conv.claim and not conv.resolved:
            claim = self._merge_claims(conv.claim, claim)
        conv.claim = claim
        conv.csp_result = None
        conv.risk_assessment = None
        conv.conversation_history.append({"role": "user", "content": user_text})
        if claim.order_id and self.csp_solver.get_order_record(claim.order_id):
            if "order_lookup" not in conv.completed_verifications:
                conv.completed_verifications.append("order_lookup")
            trace["internal_tool_calls"].append({
                "action": "LOOKUP_ORDER",
                "reason": "Look up a matching order in the demo transaction database",
                "result_summary": f"Found transaction record for {claim.order_id}",
            })
        evidence_graph_result = self.evidence_graph.search(
            claim,
            conv.completed_verifications,
            self.csp_solver.get_order_record(claim.order_id)
            if claim.order_id else None,
        )
        trace["internal_tool_calls"].append({
            "action": "SEARCH_EVIDENCE_GRAPH",
            "reason": "Match reported evidence to requirements in the applicable JSON policy",
            "result_summary": (
                f"Found {len(evidence_graph_result.required_evidence)} evidence/record "
                f"requirements across {len(evidence_graph_result.policy_ids)} applicable policies"
            ),
        })

        trace["pipeline_steps"].append({
            "step": 1,
            "module": "Module 4 (NLU)",
            "title": "Claim Extraction & Intent Recognition",
            "details": {
                "intent": claim.intent,
                "amount": claim.amount,
                "item": claim.item,
                "order_id": claim.order_id,
                "evidence_provided": claim.evidence_provided,
                "threat_level": claim.threat_level.value,
                "sentiment": claim.sentiment.value,
                "adversarial_tactics": claim.adversarial_tactics,
                "contradictory_claims": claim.contradictory_claims,
                "supporting_evidence": claim.supporting_evidence,
                "emotional_appeals": claim.emotional_appeals,
                "threats": claim.threats,
                "persuasive_strategies": claim.persuasive_strategies,
                "conversation_history": claim.conversation_history,
            }
        })

        # -----------------------------------------------------------------------
        # Step 2: RL-Centric Tool-Calling Loop (Module 4 Orchestrator)
        # -----------------------------------------------------------------------
        csp_queried = False
        bayes_queried = False
        max_internal_loops = 4
        chosen_action: Optional[ActionType] = None
        q_values_debug: Dict[str, float] = {}

        chosen_state_key = ""
        for loop_idx in range(max_internal_loops):
            # Select action using Q-learning policy with valid action masking
            state_key = self.rl_agent.discretize_state(conv, csp_queried, bayes_queried)
            action, q_vals = self.rl_agent.select_action(
                conv, csp_queried, bayes_queried, force_exploitation=True
            )
            q_values_debug = q_vals

            if action == ActionType.CALL_CSP and not csp_queried:
                # Epistemic action: Query Module 1 CSP solver
                csp_result = self.csp_solver.evaluate(
                    claim, conv.completed_verifications, evidence_graph_result
                )
                conv.csp_result = csp_result
                csp_queried = True

                trace["internal_tool_calls"].append({
                    "action": "CALL_CSP",
                    "reason": "RL Agent requested rule verification against policy graph",
                    "result_summary": csp_result.explanation
                })
                continue

            elif action == ActionType.CALL_BAYES and not bayes_queried:
                # Epistemic action: Query Module 3 Bayesian Risk Assessor
                if conv.csp_result:
                    risk_assessment = self.bayes_assessor.assess_risk(
                        claim, conv.csp_result, conv.completed_verifications
                    )
                    conv.risk_assessment = risk_assessment
                    bayes_queried = True

                    trace["internal_tool_calls"].append({
                        "action": "CALL_BAYES",
                        "reason": "RL Agent requested probabilistic risk assessment on uncertain constraints",
                        "result_summary": f"P(Fraud) = {risk_assessment.fraud_probability:.2%}, Evidence Reliability = {risk_assessment.evidence_reliability:.2%}"
                    })
                continue

            else:
                # External communicative action selected
                chosen_action = action
                chosen_state_key = state_key
                break

        # Fallback if loop ended on an internal action
        if chosen_action is None or chosen_action in (ActionType.CALL_CSP, ActionType.CALL_BAYES):
            # If not evaluated yet, run CSP and Bayes deterministically as safety guard
            if not csp_queried:
                conv.csp_result = self.csp_solver.evaluate(
                    claim, conv.completed_verifications, evidence_graph_result
                )
            if not bayes_queried and conv.csp_result and conv.csp_result.has_uncertain:
                conv.risk_assessment = self.bayes_assessor.assess_risk(
                    claim, conv.csp_result, conv.completed_verifications
                )
            chosen_action = ActionType.ASK_FOR_EVIDENCE
            chosen_state_key = self.rl_agent.discretize_state(conv, True, bayes_queried)

        conv.current_action = chosen_action
        conv.action_history.append(chosen_action.value)

        # Log CSP and Bayes steps in pipeline trace
        if conv.csp_result:
            trace["pipeline_steps"].append({
                "step": 2,
                "module": "Module 1 (CSP Rule Engine)",
                "title": "Constraint Satisfaction Evaluation",
                "details": {
                    "applicable_policies": conv.csp_result.applicable_policies,
                    "violated_policies": conv.csp_result.violated_policies,
                    "constraints": {k: v.value for k, v in conv.csp_result.constraints.items()},
                    "has_uncertain": conv.csp_result.has_uncertain,
                    "has_manipulated": conv.csp_result.has_manipulated,
                    "evidence_graph": conv.csp_result.evidence_graph,
                    "explanation": conv.csp_result.explanation
                }
            })

        if conv.risk_assessment:
            trace["pipeline_steps"].append({
                "step": 3,
                "module": "Module 3 (Bayesian Network)",
                "title": "Probabilistic Fraud & Evidence Assessment",
                "details": {
                    "fraud_probability": conv.risk_assessment.fraud_probability,
                    "evidence_reliability": conv.risk_assessment.evidence_reliability,
                    "evidence_strength": conv.risk_assessment.evidence_strength.value,
                    "risk_factors": conv.risk_assessment.risk_factors,
                    "recommendation": conv.risk_assessment.recommendation,
                    "posterior_details": conv.risk_assessment.posterior_details
                }
            })

        trace["pipeline_steps"].append({
            "step": 4,
            "module": "Module 4 (RL Agent)",
            "title": "Strategic Macro Action Selection",
            "details": {
                "selected_action": chosen_action.value,
                "q_values": {k: round(v, 2) for k, v in q_values_debug.items()},
                "state_key": self.rl_agent.discretize_state(conv, csp_queried, bayes_queried)
            }
        })

        # -----------------------------------------------------------------------
        # Step 3: Sequential Planning & Precondition Gate (Module 2)
        # -----------------------------------------------------------------------
        plan_result = self.planner.generate_plan("case_resolved", conv)
        conv.plan_result = plan_result

        trace["pipeline_steps"].append({
            "step": 5,
            "module": "Module 2 (Sequential Planning)",
            "title": "Preconditions Check & Safety Gate",
            "details": {
                "action_approved": plan_result.action_approved,
                "block_reason": plan_result.block_reason,
                "completed_steps": plan_result.completed_steps,
                "remaining_steps": plan_result.remaining_steps,
                "total_plan_steps": len(plan_result.steps),
                "plan_visualization": [
                    {"action": s.action, "completed": s.completed, "desc": s.description}
                    for s in plan_result.steps
                ]
            }
        })

        # -----------------------------------------------------------------------
        # Step 4: Minimax Tactical Conversational Defense (Module 1)
        # -----------------------------------------------------------------------
        applicable_clauses = conv.csp_result.applicable_policies if conv.csp_result else []
        minimax_result = self.minimax_engine.search(chosen_action, claim, applicable_clauses)
        conv.minimax_result = minimax_result

        trace["pipeline_steps"].append({
            "step": 6,
            "module": "Module 1 (Minimax Search)",
            "title": "Tactical Dialogue Defense (Alpha-Beta Pruning)",
            "details": {
                "selected_response_type": minimax_result.selected_response_type,
                "compliance_score": round(minimax_result.compliance_score, 3),
                "nodes_evaluated": minimax_result.evaluated_nodes,
                "branches_pruned": minimax_result.pruned_branches,
                "tactical_reasoning": minimax_result.reasoning
            }
        })

        # -----------------------------------------------------------------------
        # Step 5: Grounded Response Generation (Module 5)
        # -----------------------------------------------------------------------
        try:
            generated_resp = self.generator.generate(
                action=chosen_action,
                claim=claim,
                csp_result=conv.csp_result,
                risk_assessment=conv.risk_assessment,
                plan_result=plan_result,
                minimax_result=minimax_result
            )
        except Exception:
            fallback_gen = ResponseGenerator(provider="template")
            generated_resp = fallback_gen.generate(
                action=chosen_action,
                claim=claim,
                csp_result=conv.csp_result,
                risk_assessment=conv.risk_assessment,
                plan_result=plan_result,
                minimax_result=minimax_result
            )
        conv.conversation_history.append({"role": "assistant", "content": generated_resp.text})

        trace["pipeline_steps"].append({
            "step": 7,
            "module": "Module 5 (Ollama Response Generation)",
            "title": "Grounded Natural Language Response Generation",
            "details": {
                "provider": self.generator.provider,
                "cited_policies": generated_resp.cited_policies,
                "tone": generated_resp.tone,
                "available_actions": generated_resp.available_actions
            }
        })

        # -----------------------------------------------------------------------
        # Step 6: Reward Calculation & RL Learning Update
        # -----------------------------------------------------------------------
        prev_state_key = chosen_state_key or self.rl_agent.discretize_state(
            conv, csp_queried, bayes_queried
        )
        if learn:
            reward, breakdown = self.rl_agent.calculate_reward(
                conv, chosen_action, csp_queried, bayes_queried
            )
            conv.total_reward += reward
            next_state_key = self.rl_agent.discretize_state(
                conv, True, bayes_queried
            )
            self.rl_agent.update_q_value(
                prev_state_key, chosen_action, reward, next_state_key, done=True
            )
            self.rl_agent.save_q_table()

        is_done = chosen_action in (ActionType.APPROVE, ActionType.REJECT) and plan_result.action_approved

        # Mark terminal if resolved
        if is_done:
            conv.resolved = True
            conv.resolution = chosen_action.value

        # Return full payload for both chat interface and debugging inspector
        return {
            "session_id": conv.session_id,
            "turn": conv.turn,
            "action": chosen_action.value,
            "response": {
                "text": generated_resp.text,
                "cited_policies": generated_resp.cited_policies,
                "evidence_status": generated_resp.evidence_status,
                "available_actions": generated_resp.available_actions,
                "tone": generated_resp.tone
            },
            "claim_summary": {
                "intent": claim.intent,
                "amount": claim.amount,
                "item": claim.item,
                "threat_level": claim.threat_level.value,
                "sentiment": claim.sentiment.value,
                "evidence_provided": claim.evidence_provided,
                "adversarial_tactics": claim.adversarial_tactics
            },
            "csp_summary": {
                "satisfied": sum(1 for s in conv.csp_result.constraints.values() if s == ConstraintStatus.SATISFIED) if conv.csp_result else 0,
                "violated": sum(1 for s in conv.csp_result.constraints.values() if s == ConstraintStatus.VIOLATED) if conv.csp_result else 0,
                "uncertain": sum(1 for s in conv.csp_result.constraints.values() if s == ConstraintStatus.UNCERTAIN) if conv.csp_result else 0,
                "manipulated": sum(1 for s in conv.csp_result.constraints.values() if s == ConstraintStatus.POTENTIALLY_MANIPULATED) if conv.csp_result else 0,
                "violated_policies": conv.csp_result.violated_policies if conv.csp_result else [],
                "explanation": conv.csp_result.explanation if conv.csp_result else "",
                "evidence_graph": (
                    conv.csp_result.evidence_graph if conv.csp_result else {}
                ),
            },
            "risk_summary": {
                "fraud_probability": conv.risk_assessment.fraud_probability if conv.risk_assessment else 0.0,
                "evidence_reliability": conv.risk_assessment.evidence_reliability if conv.risk_assessment else 1.0,
                "evidence_strength": conv.risk_assessment.evidence_strength.value if conv.risk_assessment else "moderate",
                "risk_factors": conv.risk_assessment.risk_factors if conv.risk_assessment else [],
                "recommendation": conv.risk_assessment.recommendation if conv.risk_assessment else "Proceed with standard verification"
            },
            "safety_gate": {
                "approved": plan_result.action_approved,
                "block_reason": plan_result.block_reason,
                "completed_steps": plan_result.completed_steps,
                "remaining_steps": plan_result.remaining_steps
            },
            "verification_actions": len(conv.completed_verifications),
            "minimax_summary": {
                "tone": minimax_result.selected_response_type,
                "compliance_score": minimax_result.compliance_score,
                "nodes_evaluated": minimax_result.evaluated_nodes,
                "pruned_branches": minimax_result.pruned_branches
            },
            "trace": trace,
            # Raw objects for multi-agent simulation
            "claim_raw": claim,
            "csp_raw": conv.csp_result,
            "risk_raw": conv.risk_assessment
        }
