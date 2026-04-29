"""GEPA Tree Adapter — converts Lambda-RLM tree traces to GEPA reflective dataset format.

GEPA expects reflective datasets for per-node credit assignment during
prompt optimization.  This adapter's key innovation: per-node surgical
context instead of flat trace.  Each entry targets a SPECIFIC failing node
with full tree context so GEPA can do targeted optimization.
"""

from __future__ import annotations

from rsocas.contracts.evaluation import DisagreementSignal
from rsocas.contracts.traces import TreeTrace


class TreeTraceGEPAAdapter:
    """Adapts Lambda-RLM tree traces into GEPA's reflective dataset format.

    GEPA expects reflective datasets as::

        {
            "component_name": [
                {
                    "Inputs": {"field": "value"},
                    "Generated Outputs": {"field": "value"},
                    "Feedback": "textual feedback string",
                },
            ]
        }

    This adapter's key innovation: per-node surgical context instead of flat
    trace.  Each entry targets a SPECIFIC failing node with full tree context.
    """

    def identify_failing_nodes(
        self,
        trace: TreeTrace,
        disagreement: DisagreementSignal,
    ) -> list[str]:
        """Return node_ids where per-node disagreement is highest.

        Sorts by disagreement score descending.  Returns all nodes with
        nonzero disagreement.
        """
        if not disagreement.per_node:
            return []

        sorted_nodes = sorted(
            disagreement.per_node.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [node_id for node_id, score in sorted_nodes if score > 0.0]

    def extract_subtree_context(self, trace: TreeTrace, node_id: str) -> str:
        """Extract context around a failing node.

        Returns a formatted string describing the node's position in the
        execution tree, its inputs/outputs, and its relationship to siblings
        and parent.
        """
        node_map = {n.id: n for n in trace.nodes}
        node = node_map.get(node_id)
        if node is None:
            return f"Node [{node_id}] not found in trace."

        # Find the leaf trace for this node (if it is a leaf)
        leaf_map = {lt.node_id: lt for lt in trace.leaf_traces}
        leaf = leaf_map.get(node_id)

        # Find parent node (the node whose children include this node_id)
        parent = _find_parent(trace, node_id)

        # Count siblings
        sibling_count = 0
        sibling_outputs: list[str] = []
        if parent is not None:
            for child_id in parent.children:
                if child_id != node_id:
                    sibling_count += 1
                    sibling_node = node_map.get(child_id)
                    if sibling_node is not None:
                        summary = sibling_node.output[:100]
                        sibling_outputs.append(summary)

        lines: list[str] = []
        lines.append(
            f"Node [{node_id}] at depth [{node.depth}], "
            f"position [{node.position}] of [{trace.k}]."
        )
        lines.append(f"Combinator: [{node.combinator}]. Input size: [{node.input_size}] chars.")

        if sibling_outputs:
            joined = "; ".join(sibling_outputs)
            lines.append(f"Sibling nodes produced: [{joined}].")
        elif sibling_count == 0 and parent is not None:
            lines.append("No sibling nodes.")

        if leaf is not None:
            prompt_excerpt = leaf.prompt[:500]
            lines.append(f"This node received: [{prompt_excerpt}].")
        else:
            lines.append(f"This node received: [{node.output[:500]}].")

        lines.append(f"This node produced: [{node.output}].")

        if parent is not None:
            lines.append(
                f"Parent node used [{parent.combinator}] to combine results."
            )

        return "\n".join(lines)

    def make_reflective_dataset(
        self,
        trace: TreeTrace,
        disagreement: DisagreementSignal,
        component_name: str = "leaf_prompt",
    ) -> dict[str, list[dict[str, str]]]:
        """Convert a tree trace + disagreement into GEPA reflective dataset format.

        For each failing node:
        - Inputs: the chunk text and question
        - Generated Outputs: the node's output
        - Feedback: surgical context from extract_subtree_context
        """
        failing_ids = self.identify_failing_nodes(trace, disagreement)
        if not failing_ids:
            return {component_name: []}

        node_map = {n.id: n for n in trace.nodes}
        leaf_map = {lt.node_id: lt for lt in trace.leaf_traces}

        entries: list[dict[str, str]] = []
        for nid in failing_ids:
            node = node_map.get(nid)
            if node is None:
                continue

            leaf = leaf_map.get(nid)
            input_text = leaf.prompt if leaf else ""
            output_text = node.output
            feedback = self.extract_subtree_context(trace, nid)

            entries.append({
                "Inputs": {"context": input_text, "task_type": trace.task_type},
                "Generated Outputs": {"answer": output_text},
                "Feedback": feedback,
            })

        return {component_name: entries}

    def adapt_for_gepa_optimize(
        self,
        traces: list[tuple[TreeTrace, DisagreementSignal, str | None]],
        component_name: str = "leaf_prompt",
    ) -> dict[str, list[dict[str, str]]]:
        """Batch adapter: convert multiple traces into a single reflective dataset.

        Args:
            traces: List of (trace, disagreement, ground_truth) tuples.
            component_name: The GEPA component name for the dataset.

        Returns:
            Aggregated reflective dataset across all traces.
        """
        aggregated: list[dict[str, str]] = []

        for trace, disagreement, _ground_truth in traces:
            dataset = self.make_reflective_dataset(
                trace, disagreement, component_name
            )
            aggregated.extend(dataset.get(component_name, []))

        return {component_name: aggregated}


def _find_parent(trace: TreeTrace, node_id: str) -> object | None:
    """Find the parent NodeTrace whose children contain node_id."""
    for node in trace.nodes:
        if node_id in node.children:
            return node
    return None
