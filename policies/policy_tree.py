
import json
import networkx as nx
import pickle 
def build_policy_graph(json_filepath):
    # 1. Load the JSON data
    with open(json_filepath, 'r') as file:
        data = json.load(file)
       
    # Initialize the Directed Graph
    policy_graph = nx.DiGraph()
   
    # 2. Add the Master Root Node
    root_node = "Company_Root"
    policy_graph.add_node(
        root_node,
        type="root",
        reference=data.get("company_reference"),
        version=data.get("policy_version")
    )
   
    # 3. Iterate through each policy to create category branches
    for policy in data.get("policies", []):
        policy_id = policy.get("policy_id")
       
        # Add Policy Branch Node
        policy_graph.add_node(
            policy_id,
            type="policy_category",
            name=policy.get("name"),
            category=policy.get("type"),
            possible_actions=policy.get("possible_actions")
        )
       
        # Connect the policy to the master root
        policy_graph.add_edge(root_node, policy_id)
       
        # 4. Iterate through constraints to create leaf nodes
        for constraint in policy.get("constraints", []):
            constraint_id = constraint.get("constraint_id")
           
            # Add Constraint Node
            # We embed the entire raw constraint dictionary into 'csp_logic'
            # so the CSP module has access to nested conditional/all fields.
            policy_graph.add_node(
                constraint_id,
                type="clause",
                constraint_type=constraint.get("type"),
                description=constraint.get("description"),
                operator=constraint.get("operator", None),
                csp_logic=constraint
            )
           
            # Connect the constraint clause to its parent policy
            policy_graph.add_edge(policy_id, constraint_id)
           
    return policy_graph

# ==========================================
# Execution and Graph Traversal Example
# ==========================================

if __name__ == "__main__":
    # Build the graph using the provided n.json file
    graph = build_policy_graph('n.json')

    print(f"Successfully built policy graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.\n")

    # Example: Simulating the search algorithm isolating the 'Warranty Claim' constraints
    target_policy_node = "P1"
   
    print(f"Extracted Constraints for {graph.nodes[target_policy_node]['name']}:")
   
    # Traverse the successors (children) of the P1 node
    for child in graph.successors(target_policy_node):
        clause_data = graph.nodes[child]
       
        print(f" - [{child}] {clause_data['description']}")
        print(f"     Type: {clause_data['constraint_type']}")
       
        # Show how easy it is to pull the specific logic for the CSP
        if clause_data['constraint_type'] == 'comparison':
            logic = clause_data['csp_logic']
            left_field = logic.get('left', {}).get('field', 'UNKNOWN')
            operator = logic.get('operator')
           
            # Check if right side is a value or another field
            right_dict = logic.get('right', {})
            right_val = right_dict.get('value', right_dict.get('field', 'UNKNOWN'))
           
            print(f"     CSP Logic: ({left_field} {operator} {right_val})")

with open('graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

# # --- Load Graph ---
# with open('graph.pkl', 'rb') as f:
#     graph = pickle.load(f)