import json
import networkx as nx

def build_evidence_graph_and_export_csp(json_filepath, output_filepath):
    # 1. Load the case data
    with open(json_filepath, 'r') as file:
        data = json.load(file)
       
    case_id = data.get("case_id")
   
    # 2. Merge all CSP variables directly from the Evidence side
    request_data = data.get("request", {})
    claim_facts = data.get("claim_facts", {})
    derived_values = data.get("derived_values", {})
   
    # Combine them into a single flat dictionary
    final_csp_payload = {**request_data, **claim_facts, **derived_values}
   
    # 3. Initialize the Graph
    evidence_graph = nx.DiGraph(
        case_id=case_id,
        policy_type=data.get("policy_type"),
        recommended_action=data.get("recommended_action")
    )
   
    # 4. Add the Core Case Node (Attaching the fully merged CSP facts here)
    evidence_graph.add_node(
        case_id,
        type="case_root",
        intent=request_data.get("intent"),
        csp_facts=final_csp_payload
    )
   
    # 5. Add the Customer Node
    customer_info = data.get("customer", {})
    cust_id = customer_info.get("customer_id")
    if cust_id:
        evidence_graph.add_node(
            cust_id,
            type="customer",
            name=customer_info.get("name"),
            auth_status=customer_info.get("customer_authorization_status")
        )
        evidence_graph.add_edge(cust_id, case_id, relation="INITIATED_CASE")

    # 6. Add the Order/Product Nodes
    order_id = request_data.get("order_id")
    product_id = request_data.get("product_id") or request_data.get("claimed_product_id")
   
    if order_id:
        evidence_graph.add_node(order_id, type="order")
    if product_id:
        evidence_graph.add_node(product_id, type="product", reported_defect=request_data.get("reported_defect"))

    # 7. Dynamically add all Evidence Nodes
    for evidence in data.get("evidence_nodes", []):
        evidence_graph.add_node(
            evidence.get("evidence_id"),
            type="evidence",
            evidence_category=evidence.get("type"),
            source=evidence.get("source"),
            verification_status=evidence.get("verification_status"),
            attributes=evidence.get("attributes", {})
        )

    # 8. Apply all relational Edges
    for edge in data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation")
        evidence_graph.add_edge(source, target, relation=relation)

    # 9. EXPORT THE CSP PAYLOAD TO A SEPARATE FILE
    with open(output_filepath, 'w') as out_file:
        json.dump(final_csp_payload, out_file, indent=4)

    return evidence_graph, case_id

# ==========================================
# Execution and Traversal Example
# ==========================================

if __name__ == "__main__":
    # Point this to your input case JSON
    input_file = 'p1_warranty_claim.json'
   
    # The new file where the CSP dictionary will be saved
    output_file = 'csp_payload_output.json'
   
    graph, main_case_id = build_evidence_graph_and_export_csp(input_file, output_file)

    print(f"Evidence Graph built successfully for {main_case_id}!")
    print(f"Total Nodes: {graph.number_of_nodes()} | Total Edges: {graph.number_of_edges()}\n")
    print(f"--> The combined CSP payload has been exported to '{output_file}'.")