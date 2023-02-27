from time import time
from typing import Optional, List, Tuple, Dict, Type, TYPE_CHECKING

from deepproblog.semiring import Result
from deepproblog.semiring import Semiring
from problog.formula import LogicDAG, LogicFormula
from problog.logic import Term, term2list
from problog.sdd_formula import SDD

if TYPE_CHECKING:
    from deepproblog.model import Model


class ArithmeticCircuit(object):
    def __init__(
        self,
        formula: LogicFormula,
        semiring: Optional[Type[Semiring]],
        ground_time: Optional[float] = None,
    ):
        """
        :param formula: The ground logic formula that will be compiled.
        :param semiring: Factory method that creates the semiring
        :param ground_time: Optional. Keep track of time it took to ground out formula. Used for timing statistics.
        """
        start = time()
        self.proof = LogicDAG.createFrom(formula, keep_named=True)
        self.sdd = SDD.create_from(self.proof)
        self.compile_time = time() - start
        self.re_evaluate = False
        self.semiring = semiring
        self.ground_time = ground_time
        self.results = dict()

    def evaluate(
        self,
        model: "Model",
        substitution: Optional[dict] = None,
        re_evaluate: bool = True,
    ) -> Result:
        """
        Evaluates the arithmetic circuit.
        :param model: The model (neural network and parameters) to evaluate the AC in.
        :param substitution: Optional dict. The substitution applied to the parameterized AC. See apply_term.
        :param re_evaluate: Force re-evaluation of the neural networks
        :return:
        """
        substitution = dict() if substitution is None else substitution
        values = None
        if self.re_evaluate or re_evaluate:
            to_evaluate = self.extract_neural()
            to_evaluate = [(e[0], e[1].apply_term(substitution)) for e in to_evaluate]
            values = model.evaluate_nn(to_evaluate)
        self.re_evaluate = True
        semiring = self.semiring(model, substitution, values)
        start = time()
        evaluation = self.sdd.evaluate(semiring=semiring)
        if substitution is not None:
            evaluation = {k.apply_term(substitution): evaluation[k] for k in evaluation}
        eval_time = time() - start
        return Result(
            evaluation,
            semiring,
            self.ground_time,
            self.compile_time,
            eval_time,
            self.proof,
        )

    def extract_neural(self) -> List[Tuple[Term, Term]]:
        """
        :return: Returns a set of all neural predicates that need to be evaluated.
        """
        neural_eval = []
        weights = self.sdd.get_weights()
        for w in weights:
            w = weights[w]
            if type(w) is Term:
                if w.functor == "nn":
                    self._add_ordered_evaluation(w.args[0], w.args[1], neural_eval)

        return neural_eval

    def _add_ordered_evaluation(self, name, arguments, evals):
        # Check arguments for tensors that need to be evaluated
        for argument in term2list(arguments, deep=False):
            if argument.functor == "tensor":
                argument = argument.args[0]
                if argument.functor == "nn":
                    self._add_ordered_evaluation(*argument.args, evals)
        k = (name, arguments)
        if k not in evals:
            evals.append(k)

    def get_named(self) -> Dict[Term, int]:
        """
        :return: A dictionary mapping all named nodes in the SDD to their node id.
        """
        named = dict()
        for n in self.sdd.get_names():
            named[n[0]] = n[1]
        return named
