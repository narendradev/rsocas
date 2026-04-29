"""DSPy Leaf Registry — manages DSPy modules for Lambda-RLM leaf nodes.

Provides a registry of leaf module specifications that can be converted
into actual DSPy Modules when DSPy is available.  Leaf functions are
injectable into Lambda-RLM's REPL globals to route llm_query calls
through DSPy's optimized prompting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeafModuleSpec:
    """Specification for a DSPy leaf module (without requiring DSPy import).

    Attributes:
        task_type: Identifier for the task (e.g. "summarization", "qa").
        signature: DSPy signature string (e.g. "context, question -> answer").
        instruction: Optional instruction prefix for the module.
        demos: Optional list of few-shot demonstrations as dicts.
    """

    task_type: str
    signature: str = "context, question -> answer"
    instruction: str = ""
    demos: tuple[dict, ...] = ()


class DSPyLeafRegistry:
    """Registry of DSPy leaf modules for Lambda-RLM integration.

    Manages leaf module specifications and provides callables
    for injection into Lambda-RLM's REPL.
    """

    def __init__(self) -> None:
        self._specs: dict[str, LeafModuleSpec] = {}
        self._modules: dict[str, object] = {}

    def register(self, spec: LeafModuleSpec) -> None:
        """Register a leaf module specification.

        Overwrites any existing spec for the same task_type.
        """
        self._specs[spec.task_type] = spec
        # Clear cached module if spec changed
        self._modules.pop(spec.task_type, None)

    def get_leaf_fn(self, task_type: str) -> callable:
        """Get a callable function(prompt: str) -> str for REPL injection.

        If DSPy is available and a module exists: wraps the dspy.Module call.
        If DSPy is not available: returns a callable that raises ImportError.

        Raises:
            KeyError: If task_type is not registered.
        """
        if task_type not in self._specs:
            raise KeyError(f"No spec registered for task_type: {task_type!r}")

        spec = self._specs[task_type]

        # Try to get or create a DSPy module
        try:
            module = self._get_or_create_module(spec)
        except ImportError:
            def _no_dspy(prompt: str) -> str:
                raise ImportError(
                    "DSPy is not installed. "
                    "Install it with: pip install dspy-ai"
                )
            return _no_dspy

        def _call_module(prompt: str) -> str:
            result = module(context=prompt, question="")
            # DSPy modules return Prediction objects; extract the answer field
            return str(getattr(result, "answer", result))

        return _call_module

    def create_dspy_module(self, spec: LeafModuleSpec) -> object:
        """Create an actual DSPy Module from a spec.

        Requires DSPy to be installed.

        Returns:
            A ``dspy.Predict`` instance configured with the spec.

        Raises:
            ImportError: If DSPy is not installed.
        """
        try:
            import dspy  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "DSPy is not installed. Install it with: pip install dspy-ai"
            ) from exc

        predictor = dspy.Predict(spec.signature)
        if spec.instruction:
            predictor.signature = predictor.signature.with_instructions(
                spec.instruction
            )
        return predictor

    def optimize(
        self,
        task_type: str,
        trainset: list,
        metric: callable,
        optimizer_name: str = "BootstrapFewShot",
    ) -> None:
        """Optimize a leaf module using a DSPy optimizer.

        Requires DSPy to be installed.

        Args:
            task_type: The registered task type to optimize.
            trainset: Training examples for the optimizer.
            metric: Evaluation metric callable.
            optimizer_name: Name of the DSPy optimizer class.

        Raises:
            ImportError: If DSPy is not installed.
            KeyError: If task_type is not registered.
        """
        if task_type not in self._specs:
            raise KeyError(f"No spec registered for task_type: {task_type!r}")

        try:
            import dspy  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "DSPy is not installed. Install it with: pip install dspy-ai"
            ) from exc

        spec = self._specs[task_type]
        module = self._get_or_create_module(spec)

        optimizer_cls = getattr(dspy, optimizer_name, None)
        if optimizer_cls is None:
            raise ValueError(f"Unknown DSPy optimizer: {optimizer_name!r}")

        optimizer = optimizer_cls(metric=metric)
        optimized = optimizer.compile(module, trainset=trainset)
        self._modules[task_type] = optimized

    def inject_into_repl(self, repl_globals: dict, task_type: str) -> None:
        """Replace ``llm_query`` in repl globals with a DSPy module call.

        This is how Lambda-RLM leaf nodes become DSPy Modules:
        the generated Phi code calls ``llm_query(prompt)``, which now
        routes through our DSPy module instead of raw LLM.

        Args:
            repl_globals: The globals dict for the REPL environment.
            task_type: The registered task type whose leaf function to inject.
        """
        leaf_fn = self.get_leaf_fn(task_type)
        repl_globals["llm_query"] = leaf_fn

    def list_registered(self) -> list[str]:
        """List all registered task types."""
        return sorted(self._specs.keys())

    def _get_or_create_module(self, spec: LeafModuleSpec) -> object:
        """Get cached module or create a new one.

        Raises:
            ImportError: If DSPy is not installed.
        """
        if spec.task_type in self._modules:
            return self._modules[spec.task_type]

        module = self.create_dspy_module(spec)
        self._modules[spec.task_type] = module
        return module
