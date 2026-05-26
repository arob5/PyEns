"""EnsembleRunner: drives a model over an EnsembleSpec using a Backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pyens.backends.base import Backend
from pyens.result import EnsembleResult, RunRecord
from pyens.spec import EnsembleSpec


class EnsembleRunner:
    """Drives a model callable over all runs in an ``EnsembleSpec``.

    ``EnsembleRunner`` is the bridge between the specification layer
    (``EnsembleSpec``) and the execution layer (``Backend``). It iterates the
    spec, hands the run inputs to the backend, and assembles the results into
    an :class:`~pyens.result.EnsembleResult`.

    Args:
        model: The model callable. Must accept the field names from the spec
            as keyword arguments: ``model(**inputs)`` where ``inputs`` is one
            ``(field_name → value)`` dict per run. The return value can be
            anything — a scalar, a dict, a dataframe.
        backend: The execution backend. Controls whether runs execute
            sequentially, in parallel processes, or on a remote cluster.

    Example::

        from pyens import Axis, EnsembleSpec, Fixed, Grid
        from pyens.backends import SequentialBackend
        from pyens.runner import EnsembleRunner

        def my_model(parameters, forcing):
            ...

        sites = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={
            "parameters": Fixed(base_params),
            "forcing":    Grid(forcing_list, along=sites),
        })

        runner = EnsembleRunner(my_model, SequentialBackend())
        result = runner.run(spec)
    """

    def __init__(self, model: Callable[..., Any], backend: Backend) -> None:
        self._model = model
        self._backend = backend

    def run(self, spec: EnsembleSpec) -> EnsembleResult:
        """Execute the model for every run in *spec*.

        Args:
            spec: A fully-bound ``EnsembleSpec`` (not a ``BoundSpec``).
                Call ``bound_spec(field=value)`` first if you have a
                ``BoundSpec``.

        Returns:
            An :class:`~pyens.result.EnsembleResult` with one
            :class:`~pyens.result.RunRecord` per run, in ``iter_runs()``
            order. Failed runs store the exception in place rather than
            interrupting the ensemble.
        """
        pairs = list(spec.iter_runs())
        inputs_list = [inputs for inputs, _ in pairs]
        coordinates = [coord for _, coord in pairs]

        raw = self._backend.map(self._model, inputs_list)

        records = [
            RunRecord(coordinate=coord, output=output)
            for coord, output in zip(coordinates, raw)
        ]
        return EnsembleResult(records)
