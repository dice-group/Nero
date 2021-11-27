import torch
from torch import nn
from typing import Dict, List, Iterable
from ontolearn.search import RL_State
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLClass, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression
from ontolearn.refinement_operators import LengthBasedRefinement
from .static_funcs import apply_rho_on_rl_state
import time


class NCEL:
    def __init__(self, model: torch.nn.Module,
                 quality_func,
                 target_class_expressions,
                 instance_idx_mapping: Dict):
        self.model = model
        self.quality_func = quality_func
        self.target_class_expressions = target_class_expressions
        self.instance_idx_mapping = instance_idx_mapping
        self.inverse_instance_idx_mapping=dict(zip(self.instance_idx_mapping.values(),self.instance_idx_mapping.keys()))
        self.renderer = DLSyntaxObjectRenderer()
        self.max_top_k = len(self.target_class_expressions)

    def forward(self, *, xpos, xneg):
        return self.model(xpos, xneg)

    def positive_embeddings_from_iterable_of_individuals(self, pos: Iterable[str]):
        raise NotImplementedError()
        """
        pred = self.forward(xpos=torch.LongTensor([[self.instance_idx_mapping[i] for i in pos]]),
                            xneg=torch.LongTensor([[self.instance_idx_mapping[i] for i in neg]]))
        self.model(xpos, xneg)
        """
    def negative_embeddings(self, xpos):
        return self.model(xpos, xneg)

    def __intersection_topK(self, results, set_pos, set_neg):
        """
        Intersect top K class expressions

        This often deteriorates the performance. This may indicate that topK concepts explain the different aspect of
        the goal expression
        :param results:
        :param set_pos:
        :param set_neg:
        :return:
        """
        # apply some operation
        for (_, exp_i) in results:
            for (__, exp_j) in results:
                if exp_i == exp_j:
                    continue

                next_rl_state = RL_State(OWLObjectIntersectionOf((exp_i.concept, exp_j.concept)), parent_node=exp_i)
                next_rl_state.length = self.kb.cl(next_rl_state.concept)
                next_rl_state.instances = set(self.kb.individuals(next_rl_state.concept))
                quality = self.quality_func(
                    instances={i.get_iri().as_str() for i in next_rl_state.instances},
                    positive_examples=set_pos, negative_examples=set_neg)
                # Do not assing quality for target states
                print(exp_i)
                print(exp_j)
                print(quality)

    def __union_topK(self, results, set_pos, set_neg):
        """
        Union topK expressions
        :param results:
        :param set_pos:
        :param set_neg:
        :return:
        """
        # apply some operation
        for (_, exp_i) in results:
            for (__, exp_j) in results:
                if exp_i == exp_j:
                    continue

                next_rl_state = RL_State(OWLObjectUnionOf((exp_i.concept, exp_j.concept)), parent_node=exp_i)
                next_rl_state.length = self.kb.cl(next_rl_state.concept)
                next_rl_state.instances = set(self.kb.individuals(next_rl_state.concept))
                quality = self.quality_func(
                    instances={i.get_iri().as_str() for i in next_rl_state.instances},
                    positive_examples=set_pos, negative_examples=set_neg)

    def __refine_topK(self, results, set_pos, set_neg, stop_at):
        extended_results = []
        for ith, (_, topK_target_expression) in enumerate(results):
            for refinement_topK in apply_rho_on_rl_state(topK_target_expression, self.rho, self.kb):
                s: float = self.quality_func(
                    instances={i.get_iri().as_str() for i in refinement_topK.instances},
                    positive_examples=set_pos, negative_examples=set_neg)
                if s > _:
                    # print(f'Refinement ({s}) is better than its parent ({_})')
                    extended_results.append((s, refinement_topK))
                    if s == 1.0:
                        print('Goal found in the local search')
                        break
                if ith == stop_at:
                    break
        return extended_results

    def fit(self, pos: [str], neg: [str], topK: int, local_search=False) -> Dict:
        if topK is None:
            topK=self.max_top_k
        try:
            assert topK > 0
        except AssertionError:
            print(f'topK must be greater than 0. Currently:{topK}')
            topK=self.max_top_k
        start_time = time.time()
        goal_found = False
        try:
            idx_pos = [self.instance_idx_mapping[i] for i in pos]
        except KeyError:
            print('Ensure that Positive examples can be found in the input KG')
            print(pos)
            exit(1)
        try:
            idx_neg = [self.instance_idx_mapping[i] for i in neg]
        except KeyError:
            print('Ensure that Positive examples can be found in the input KG')
            print(neg)
            exit(1)
        pred = self.forward(xpos=torch.LongTensor([idx_pos]),
                            xneg=torch.LongTensor([idx_neg]))

        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        results = []
        # We could apply multi_processing here
        # Explore only top K class expressions that have received highest K scores
        set_str_pos = set(pos)
        set_str_neg = set(neg)

        for i in sort_idxs[:topK]:
            str_instances={ self.inverse_instance_idx_mapping[i] for i in self.target_class_expressions[i].idx_individuals}
            s: float = self.quality_func(
                instances=str_instances,
                positive_examples=set_str_pos, negative_examples=set_str_neg)
            results.append((s, self.target_class_expressions[i],str_instances))
            if s == 1.0:
                # print('Goal Found in the tunnelling')
                goal_found = True
                break
        if goal_found is False and local_search:
            extended_results = self.__refine_topK(results, set_pos, set_neg, stop_at=topK)
            results.extend(extended_results)

        num_expression_tested = len(results)
        results = sorted(results, key=lambda x: x[0], reverse=True)
        f1, top_pred, top_str_instances= results[0]

        report = {'Prediction': top_pred.name,
                  'Instances': top_str_instances,
                  'F1-Score': f1,
                  'NumClassTested': num_expression_tested,
                  'Runtime': time.time() - start_time,
                  }

        return report

    def __predict_sanity_checking(self, pos: [str], neg: [str], topK: int = None, local_search=False):
        if topK is None:
            topK = self.max_top_k
        elif isinstance(topK, int) or isinstance(topK, flat):
            try:
                assert topK > 0
                topK = int(round(topK))
            except AssertionError:
                print(f'topK must be greater than 0. Currently:{topK}')
                topK = self.max_top_k

        assert len(pos) > 0
        assert len(neg) > 0

    def predict(self, pos: [str], neg: [str], topK: int = None, local_search=False) -> List:
        start_time = time.time()
        self.__predict_sanity_checking(pos=pos, neg=neg, topK=topK, local_search=local_search)

        idx_pos = []
        idx_neg = []
        try:
            idx_pos = [self.instance_idx_mapping[i] for i in pos]
            idx_neg = [self.instance_idx_mapping[i] for i in neg]
        except KeyError:
            print('Ensure that URIs are valid and can be found in the input KG')
            print(pos)
            print(neg)
            exit(1)
        pred = self.forward(xpos=torch.LongTensor([idx_pos]),
                            xneg=torch.LongTensor([idx_neg]))

        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        results = []
        # We could apply multi_processing here
        # Explore only top K class expressions that have received highest K scores
        set_pos = set(pos)
        set_neg = set(neg)
        for the_exploration, idx_target in enumerate(sort_idxs[:topK]):
            str_instances={ self.inverse_instance_idx_mapping[_] for _ in self.target_class_expressions[idx_target].idx_individuals}

            s: float = self.quality_func(
                instances=str_instances,
                positive_examples=set_pos, negative_examples=set_neg)
            results.append((s, self.target_class_expressions[idx_target], str_instances,the_exploration+1))
            if s == 1.0:
                # if goal found break it.
                break

        return sorted(results, key=lambda x: x[0], reverse=True), time.time() - start_time

    def __str__(self):
        return f'NCEL with {self.model.name}'

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def state_dict(self):
        return self.model.state_dict()

    def parameters(self):
        return self.model.parameters()

    def embeddings_to_numpy(self):
        return self.model.embeddings.weight.data.detach().numpy()

    def get_target_class_expressions(self):
        return (self.renderer.render(cl.concept) for cl in self.target_class_expressions)