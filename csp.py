import pickle
import json
import operator
import pandas as pd
import networkx as nx

def evaluate_csp(graph_file, payload_file):
    # 1. Map string operators to Python mathematical operators
    ops = {
        '==': operator.eq, 
        '!=': operator.ne,
        '<=': operator.le, 
        '>=': operator.ge,
        '<': operator.lt,  
        '>': operator.gt,
        'in': lambda a, b: a in b,
    }
    
    # 2. Load the Policy Graph and Evidence Payload
    with open(graph_file, 'rb') as f:
        policy_graph = pickle.load(f)
        
    with open(payload_file, 'r') as f:
        payload = json.load(f)
        
    intent = payload.get("intent")
    
    # 3. Find the target Root Node for the policy category (e.g., P1 for warranty_claim)
    target_policy = None
    for node, data in policy_graph.nodes(data=True):
        if data.get('category') == intent:
            target_policy = node
            break
            
    if not target_policy:
        return f"Policy category for intent '{intent}' not found."
        
    # 4. Retrieve all sequential clauses branching from the target policy
    clauses = [n for n in policy_graph.successors(target_policy)]
    results = []
    
    # 5. Evaluate each constraint based on its type
    for clause_id in sorted(clauses, key=lambda x: int(x.split('_L')[-1])):
        clause_data = policy_graph.nodes[clause_id]
        csp_logic = clause_data.get('csp_logic', {})
        constraint_type = clause_data.get('constraint_type')
        description = clause_data.get('description', 'Unknown')
        
        status = "Uncertain"
        reason = ""
        
        # Handle mathematically comparing two values
        if constraint_type == 'comparison':
            left_field = csp_logic['left']['field']
            op_str = csp_logic['operator']
            right_val_def = csp_logic['right']
            
            # The right value might be a static target or another dynamic variable
            if 'value' in right_val_def:
                right_val = right_val_def['value']
            elif 'field' in right_val_def:
                right_val = payload.get(right_val_def['field'])
            else:
                right_val = None
                
            left_val = payload.get(left_field)
            
            # Check for missing variables
            if left_val is None or right_val is None:
                status = "Uncertain"
                reason = f"Missing data: left={left_val}, right={right_val}"
            else:
                try:
                    # Apply evaluation
                    if ops[op_str](left_val, right_val):
                        status = "Satisfied"
                    else:
                        status = "Violated"
                    reason = f"({left_val}) {op_str} ({right_val})"
                except Exception as e:
                    status = "Error"
                    reason = str(e)
                    
        # Handle checking if required evidence/data is present
        elif constraint_type == 'presence':
            left_field = csp_logic['field']
            op_str = csp_logic['operator']
            val = payload.get(left_field)
            
            if op_str == 'is_not_null':
                status = "Satisfied" if val is not None else "Uncertain"
            elif op_str == 'is_not_empty':
                status = "Satisfied" if bool(val) else "Uncertain"
            reason = f"Extracted: '{val}'"
            
        results.append({
            "Clause": clause_id,
            "Description": description,
            "Status": status,
            "Details": reason
        })
        
    return results

# Execute the evaluator and print out an organized table
results = evaluate_csp('graph.pkl', 'csp_payload_output.json')
df = pd.DataFrame(results)
print(df.to_string(index=False))