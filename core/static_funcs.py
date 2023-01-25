import itertools
from typing import List, Tuple, Set, Dict
from ontolearn.search import RL_State
import datetime
import logging
import os
import time
import torch
import json
from collections import deque
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom
import random
from random import randint
import resource
from sys import getsizeof, stderr
from itertools import chain
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

import torch
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
import gc
from .expression import TargetClassExpression, ClassExpression, UniversalQuantifierExpression, \
    ExistentialQuantifierExpression, Role
from .refinement_operator import SimpleRefinement
import time
import functools
import psutil

def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        print(f'{func.__name__} func. being computed...')
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Computing {func.__name__} took {total_time:.4f} seconds | Current Memory Usage {psutil.Process(os.getpid()).memory_info().rss / 1000000: .5} in MB')
        return result

    return timeit_wrapper

def single_universal_quantifier(*, kb, filler, atomic_look_up, role_look_up):
    renderer = DLSyntaxObjectRenderer()
    for owl_class_instance in kb.most_general_universal_restrictions(domain=kb.thing, filler=filler):

        str_role = renderer.render(owl_class_instance._property)

        if str_role not in role_look_up:
            role_look_up[str_role] = Role(name=str_role)

        filler = renderer.render(owl_class_instance._filler)
        concept = atomic_look_up[filler]

        yield UniversalQuantifierExpression(name=renderer.render(owl_class_instance),
                                            role=role_look_up[str_role],
                                            filler=concept,
                                            str_individuals=set(
                                                _.get_iri().as_str() for _ in kb.individuals(owl_class_instance)),
                                            expression_chain=['Thing'])


def single_existential_quantifiers(*, kb, filler, atomic_look_up, role_look_up):
    renderer = DLSyntaxObjectRenderer()
    for owl_class_instance in kb.most_general_existential_restrictions(domain=kb.thing, filler=filler):

        str_role = renderer.render(owl_class_instance._property)

        if str_role not in role_look_up:
            role_look_up[str_role] = Role(name=str_role)

        filler = renderer.render(owl_class_instance._filler)
        concept = atomic_look_up[filler]

        yield ExistentialQuantifierExpression(name=renderer.render(owl_class_instance), role=role_look_up[str_role],
                                              filler=concept,
                                              str_individuals=set(
                                                  _.get_iri().as_str() for _ in kb.individuals(owl_class_instance)),
                                              expression_chain=['Thing'])


def get_all_universal_quantifiers(kb, atomic_expressions):
    atomic_look_up = {i.name: i for i in atomic_expressions}
    role_look_up = dict()
    for c in atomic_expressions:
        yield from single_universal_quantifier(kb=kb, filler=c.owl_class,
                                               atomic_look_up=atomic_look_up,
                                               role_look_up=role_look_up)


def get_all_existential_quantifiers(kb, atomic_expressions):
    atomic_look_up = {i.name: i for i in atomic_expressions}
    role_look_up = dict()
    for c in atomic_expressions:
        yield from single_existential_quantifiers(kb=kb, filler=c.owl_class,
                                                  atomic_look_up=atomic_look_up,
                                                  role_look_up=role_look_up)


def get_all_quantifiers(kb, atomic_expressions):
    dummy = {i for i in atomic_expressions}
    renderer = DLSyntaxObjectRenderer()

    dummy.add(ClassExpression(name=renderer.render(kb.thing),
                              str_individuals=set(_.get_iri().as_str() for _ in kb.individuals(kb.thing)),
                              owl_class=kb.thing,
                              expression_chain=['non']))
    return get_all_universal_quantifiers(kb, dummy), get_all_existential_quantifiers(kb, dummy)

@timeit
def select_target_expressions(kb, args):
    # (1) Individual to integer mapping
    instance_idx_mapping = {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
    number_of_target_expressions = args.number_of_target_expressions
    rho = None
    # (2) Target Expression selection
    if args.target_expression_selection == 'diverse_target_expression_selection':
        target_class_expressions = diverse_target_expression_selection(kb,
                                                                       args['tolerance_for_search_unique_target_exp'],
                                                                       number_of_target_expressions,
                                                                       instance_idx_mapping)
    elif args.target_expression_selection == 'random_target_expression_selection':
        target_class_expressions = random_target_expression_selection(kb,
                                                                      number_of_target_expressions,
                                                                      instance_idx_mapping)
    elif args.target_expression_selection == 'uncorrelated_target_expression_selection':
        target_class_expressions, rho = uncorrelated_target_expression_selection(kb,
                                                                                 number_of_target_expressions,
                                                                                 instance_idx_mapping)
    else:
        raise KeyError(f'target_expression_selection:{args.target_expression_selection}')
    return target_class_expressions, instance_idx_mapping, rho


def target_expressions_binary_selection_via_refining_top(rho, kb, number_of_target_expressions, num_of_all_individuals,
                                                         instance_idx_mapping):
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.concept_len(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    renderer = DLSyntaxObjectRenderer()
    target_class_expressions = set()
    target_idx_instance_set = set()
    quantifiers = set()

    # (1) Refine Top concept
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        # (2) Continue only concept is not empty.
        if num_of_all_individuals > len(i.instances) > 0:
            # (2.1) Add OWL class expression if, its instances is not already seen
            poss_target_idx_individuals = set(instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances)
            if poss_target_idx_individuals not in target_idx_instance_set:
                # (2.2.) Add instances
                target_idx_instance_set.add(frozenset(poss_target_idx_individuals))
                # (2.3.) Create an instance
                target = TargetClassExpression(
                    label_id=len(target_idx_instance_set),
                    name=renderer.render(i.concept),
                    # str_individuals=None,
                    idx_individuals=poss_target_idx_individuals,
                    length=kb.concept_len(i.concept),
                    expression_chain=[renderer.render(x.concept) for x in retrieve_concept_chain(i)]
                )
                # (2.4) Store target
                target_class_expressions.add(target)

            # (2.4) Store for later refinement if concept is \forall or \exists
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            # (2.5) Break if cretarion met.
            if len(target_class_expressions) == number_of_target_expressions:
                break
    return target_class_expressions, target_idx_instance_set, quantifiers


def refine_selected_expressions(rho, kb, quantifiers, target_class_expressions, target_idx_instance_set,
                                tolerance_for_search_unique_target_exp, instance_idx_mapping,
                                number_of_target_expressions, num_of_all_individuals) -> None:
    renderer = DLSyntaxObjectRenderer()
    if len(target_class_expressions) < number_of_target_expressions:
        for selected_states in quantifiers:
            if len(target_class_expressions) >= number_of_target_expressions:
                break
            not_added = 0
            for ref_selected_states in apply_rho_on_rl_state(selected_states, rho, kb):
                if not_added == tolerance_for_search_unique_target_exp:
                    break
                if num_of_all_individuals > len(ref_selected_states.instances) > 0:
                    # () Check whether we have enough target class expressions
                    if len(target_class_expressions) >= number_of_target_expressions:
                        break
                    # (3.1) Add OWL class expresssion if, its instances is not already seen
                    # poss_target_individuals = frozenset(_.get_iri().as_str() for _ in ref_selected_states.instances)
                    poss_target_idx_individuals = set(
                        instance_idx_mapping[_.get_iri().as_str()] for _ in ref_selected_states.instances)
                    if poss_target_idx_individuals not in target_idx_instance_set:
                        # (3.1.) Add instances
                        target_idx_instance_set.add(frozenset(poss_target_idx_individuals))
                        # ( 3.2.) Create an instance
                        target = TargetClassExpression(
                            label_id=len(target_idx_instance_set),
                            name=renderer.render(ref_selected_states.concept),
                            idx_individuals=poss_target_idx_individuals,
                            expression_chain=[renderer.render(x.concept) for x in
                                              retrieve_concept_chain(ref_selected_states)],
                            length=kb.concept_len(ref_selected_states.concept)
                        )
                        # Add the created instance
                        target_class_expressions.add(target)
                    else:
                        not_added += 1
                else:
                    not_added += 1
                if len(target_class_expressions) >= number_of_target_expressions:
                    break
            if len(target_class_expressions) >= number_of_target_expressions:
                break

    gc.collect()


def intersect_and_union_expressions_from_iterable(target_class_expressions, target_idx_instance_set,
                                                  number_of_target_expressions):
    while len(target_idx_instance_set) < number_of_target_expressions:

        res = set()
        for i in target_class_expressions:
            for j in target_class_expressions:

                if i == j:
                    continue

                i_and_j = i * j
                if i_and_j.size > 0 and (i_and_j.idx_individuals not in target_idx_instance_set):
                    res.add(i_and_j)
                    target_idx_instance_set.add(frozenset(i_and_j.idx_individuals))
                    i_and_j.label_id = len(target_idx_instance_set)
                else:
                    del i_and_j

                if len(target_idx_instance_set) >= number_of_target_expressions:
                    break

                i_or_j = i + j
                if i_or_j.size > 0 and (i_or_j.idx_individuals not in target_idx_instance_set):
                    res.add(i_or_j)
                    target_idx_instance_set.add(frozenset(i_or_j.idx_individuals))
                    i_or_j.label_id = len(target_idx_instance_set)
                else:
                    del i_or_j

                if len(target_idx_instance_set) >= number_of_target_expressions:
                    break
        target_class_expressions.update(res)


def construct_informative_expressions(top_refinements: Set, number_of_target_expressions=None):
    """
    :param top_refinements:
    :param number_of_target_expressions:
    :return:
    """
    informative_targets = set()
    target_individuals = {frozenset(i.str_individuals) for i in top_refinements}
    for i in top_refinements:
        if len(informative_targets) == number_of_target_expressions:
            break
        for j in top_refinements:
            if i == j:
                continue
            candidate_individuals = i.str_individuals.intersection(j.str_individuals)
            if len(candidate_individuals) > 0 and (candidate_individuals not in target_individuals):
                constructed = i * j
                target_individuals.add(frozenset(constructed.str_individuals)), informative_targets.add(constructed)
                if len(informative_targets) == number_of_target_expressions:
                    break
            else:
                """ Not informative Intersection """

            candidate_individuals = i.str_individuals.union(j.str_individuals)
            if len(candidate_individuals) > 0 and (candidate_individuals not in target_individuals):
                constructed = i + j
                target_individuals.add(frozenset(constructed.str_individuals)), informative_targets.add(constructed)
                if len(informative_targets) == number_of_target_expressions:
                    break
            else:
                """ Not informative Union """
        if len(informative_targets) == number_of_target_expressions:
            break
    return informative_targets


def reduce_redundancy(sequence_of_exp) -> Set:
    targets = set()
    targets_indv = set()
    for i in sequence_of_exp:
        if len(i.str_individuals) == 0:
            continue

        if i.str_individuals in targets_indv:
            continue
        targets_indv.add(frozenset(i.str_individuals)), targets.add(i)

    del targets_indv
    gc.collect()
    return {i for i in sorted(targets, key=lambda x: x.length)}


def uncorrelated_target_expression_selection(kb, number_of_target_expressions,
                                             instance_idx_mapping):
    """
    (1) Refine Top expression and obtain all possible ALC expressions up to length 3
    (1.1) Consider only those expression as labels whose set of individuals has not been seen before
    (1.2.) E.g. {{....}, {.}, {...}}. Only  consider those expressions as labels that do not cover all individuals
    (2)
    """
    rho = SimpleRefinement(knowledge_base=kb)
    num_of_all_individuals = kb.individuals_count()
    assert len(instance_idx_mapping) == num_of_all_individuals
    nc = rho.atomic_class_expressions()
    neg_nc = rho.negated_named_class_expressions()
    q = rho.all_quantifiers()
    uncorrelated_refinements = reduce_redundancy(nc + neg_nc + q)
    #print(f'{len(uncorrelated_refinements)} number of target expressions are obtained from the atomic expressions, negations and quantifiers')
    del nc, neg_nc, q
    gc.collect()
    if len(uncorrelated_refinements) < number_of_target_expressions:
        n = number_of_target_expressions - len(uncorrelated_refinements)
        while n != 0:
            result = construct_informative_expressions(uncorrelated_refinements, n)
            uncorrelated_refinements.update(result)
            n = number_of_target_expressions - len(uncorrelated_refinements)
    else:
        """ do nothing """
    #print(f'{len(uncorrelated_refinements)} number of target expressions are obtained from intersecting and union of the atomic expressions ,negations and quantifiers')

    assert uncorrelated_refinements == reduce_redundancy(uncorrelated_refinements)
    uncorrelated_refinements = sorted(uncorrelated_refinements, key=lambda x: x.length)
    result = []
    for ith, ce in enumerate(uncorrelated_refinements):
        ce.label_id = ith
        ce.idx_individuals = set(instance_idx_mapping[i] for i in ce.str_individuals)
        result.append(ce)
    gc.collect()
    #print(f'{len(result)} number of target expressions are obtained.')
    assert len(result) >= number_of_target_expressions
    return result, rho

@timeit
def diverse_target_expression_selection(kb, tolerance_for_search_unique_target_exp, number_of_target_expressions,
                                        instance_idx_mapping) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    (1) Refine Top expression and obtain all possible ALC expressions up to length 3
    (1.1) Consider only those expression as labels whose set of individuals has not been seen before
    (1.2.) E.g. {{....}, {.}, {...}}. Only  consider those expressions as labels that do not cover all individuals
    (2)
    Select Target Expression
    :return:
    """
    # (1) Initialize a length base refinement operator
    rho = LengthBasedRefinement(knowledge_base=kb)
    num_of_all_individuals = kb.individuals_count()
    assert len(instance_idx_mapping) == num_of_all_individuals
    # (2) Obtain some target expressions via refining the most general concept, \top.
    target_class_expressions, target_idx_instance_set, quantifiers = target_expressions_binary_selection_via_refining_top(
        rho=rho,
        kb=kb,
        number_of_target_expressions=number_of_target_expressions,
        num_of_all_individuals=num_of_all_individuals,
        instance_idx_mapping=instance_idx_mapping)
    print(f'{len(target_class_expressions)} number of target expressions are obtained from the most general expression.')
    assert len(target_idx_instance_set) == len(target_class_expressions)
    # (3) Refine refinements of \top further.
    refine_selected_expressions(rho, kb, quantifiers, target_class_expressions, target_idx_instance_set,
                                tolerance_for_search_unique_target_exp, instance_idx_mapping,
                                number_of_target_expressions, num_of_all_individuals)
    print(
        f'{len(target_class_expressions)} number of target expressions are obtained from the most general expression and quantifiers')
    assert len(target_idx_instance_set) == len(target_class_expressions)
    # (4) Intersect them.
    intersect_and_union_expressions_from_iterable(target_class_expressions, target_idx_instance_set,
                                                  number_of_target_expressions)
    print(
        f'{len(target_class_expressions)} number of target expressions are obtained from the most general expression, quantifiers, and intersect/union all previous expressions')
    assert len(target_idx_instance_set) == len(target_class_expressions)
    # (5) Add an id to a target expression/ a label.

    result = []
    for ith, tce in enumerate(target_class_expressions):
        tce.label_id = ith
        result.append(tce)
    gc.collect()
    return sorted(result, key=lambda x: x.length)

@timeit
def random_target_expression_selection(kb, number_of_target_expressions, instance_idx_mapping, logger) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    Select Target Expression
    :return:
    """
    # (1) Preparation
    renderer = DLSyntaxObjectRenderer()
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.str_individuals(rl_state.concept))
    target_class_expressions.add(rl_state)
    quantifiers = set()

    rho = LengthBasedRefinement(knowledge_base=kb)
    # (2) Refine Top concept
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        # (3) Store a class expression has indv.
        if len(i.instances) > 0:
            target_class_expressions.add(i)
            # (4) Store for later refinement if concept is \forall or \exists
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                print(f'{number_of_target_expressions} target expressions generated')
                break
    # (5) Refine
    if len(target_class_expressions) < number_of_target_expressions:
        for selected_states in quantifiers:
            if len(target_class_expressions) == number_of_target_expressions:
                break
            for ref_selected_states in apply_rho_on_rl_state(selected_states, rho, kb):
                if len(ref_selected_states.instances) > 0:
                    if len(target_class_expressions) == number_of_target_expressions:
                        break
                    target_class_expressions.add(ref_selected_states)
    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort targets w.r.t. their lenghts
    # Store all target instances
    # These computation can be avoided via Priorty Queue above
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)
    labels = []
    for id_t, i in enumerate(target_class_expressions):
        target = TargetClassExpression(
            label_id=id_t,
            name=renderer.render(i.concept),
            idx_individuals=set(instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances),
            expression_chain=[renderer.render(x.concept) for x in retrieve_concept_chain(i)]
        )
        labels.append(target)
    return labels


def generate_learning_problems_from_targets(target_class_expressions: List[TargetClassExpression],
                                            instance_idx_mapping: Dict,
                                            args: Dict) -> Tuple[List[int], List[int]]:
    """
    Sample pos from targets

    :param target_class_expressions:
    :param instance_idx_mapping:
    :param args:
    :return:
    """
    print('Learning Problems are being sampled from Targets')
    instances_idx_list = list(instance_idx_mapping.values())

    pos_examples = []
    neg_examples = []
    num_individual_per_example = args.num_individual_per_example
    n = 0
    for i in range(args.num_of_learning_problems_training):
        for tce in target_class_expressions:
            try:
                pos_examples.append(random.choices(list(tce.idx_individuals), k=num_individual_per_example))
                neg_examples.append(random.choices(instances_idx_list, k=num_individual_per_example))
            except:
                # Ignore this exp
                n += 1
                continue

    assert len(pos_examples) == len(neg_examples)
    return pos_examples, neg_examples


def generate_random_learning_problems(instance_idx_mapping: Dict,
                                      args: Dict) -> Tuple[List[int], List[int]]:
    """
    Generate Learning problems
    :param instance_idx_mapping:
    :param args: hyperparameters
    :return: a list of ordered learning problems. Each inner list contains same amount of positive and negative
     examples
    """
    instances_idx_list = list(instance_idx_mapping.values())

    pos_examples = []
    neg_examples = []
    num_individual_per_example = args['num_individual_per_example']
    for i in range(args['num_of_learning_problems_training']):
        # Varianable length
        # pos_examples.append(random.choices(instances_idx_list, k=randint(1, max_num_individual_per_example)))
        # neg_examples.append(random.choices(instances_idx_list, k=randint(1, max_num_individual_per_example)))
        pos_examples.append(random.choices(instances_idx_list, k=num_individual_per_example))
        neg_examples.append(random.choices(instances_idx_list, k=num_individual_per_example))
    assert len(pos_examples) == len(neg_examples)
    return pos_examples, neg_examples


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb ''' % (point, usage[0], usage[1], usage[2] / 1024.0)


def save_as_json(*, storage_path=None, obj=None, name=None):
    with open(storage_path + f'/{name}.json', 'w') as file_descriptor:
        json.dump(obj, file_descriptor, indent=3)


def apply_rho_on_rl_state(rl_state, rho, kb):
    for i in rho.downward_refine(rl_state.concept):
        next_rl_state = RL_State(i, parent_node=rl_state)
        next_rl_state.length = kb.concept_len(next_rl_state.concept)
        next_rl_state.instances = set(kb.individuals(next_rl_state.concept))
        yield next_rl_state


def generate_target_class_expressions(lpg, kb, args):
    renderer = DLSyntaxObjectRenderer()

    number_of_target_expressions = args['number_of_target_expressions']
    # Generate target_class_expressions
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.concept_len(kb.thing)
    rl_state.instances = set(kb.str_individuals(rl_state.concept))
    target_class_expressions.add(rl_state)
    quantifiers = set()
    for i in lpg.apply_rho_on_rl_state(rl_state):
        if len(i.instances) > 0:
            target_class_expressions.add(i)
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                break
    for selected_states in quantifiers:
        for ref_selected_states in lpg.apply_rho_on_rl_state(selected_states):
            if len(ref_selected_states.instances) > 0:
                target_class_expressions.add(ref_selected_states)
        if len(target_class_expressions) == number_of_target_expressions:
            break

    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort it for the convenience. Not a must ALC formulas
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)

    # All instances belonging to targets
    target_individuals: List[Set[str]] = [{i.get_iri().as_str() for i in s.instances} for s in
                                          target_class_expressions]
    target_class_expressions: List[str] = [renderer.render(i.concept) for i in target_class_expressions]

    return target_class_expressions, target_individuals


def compute_f1_target(target_class_expressions, pos, neg):
    pos = set(pos)
    neg = set(neg)
    return [f_measure(instances=t.idx_individuals, positive_examples=pos, negative_examples=neg) for t in
            target_class_expressions]


def f_measure(*, instances: Set, positive_examples: Set, negative_examples: Set):
    tp = len(positive_examples.intersection(instances))
    # tn = len(learning_problem.kb_neg.difference(instances))

    fp = len(negative_examples.intersection(instances))
    fn = len(positive_examples.difference(instances))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0

    if precision == 0 or recall == 0:
        return 0.0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1


def accuracy(*, instances: Set, positive_examples: Set, negative_examples: Set):
    tp = len(positive_examples.intersection(instances))
    tn = len(negative_examples.difference(instances))

    fp = len(negative_examples.intersection(instances))
    fn = len(positive_examples.difference(instances))

    return (tp + tn) / (tp + tn + fp + fn)


def retrieve_concept_chain(rl_state: RL_State) -> List[RL_State]:
    hierarchy = deque()
    if rl_state.parent_node:
        hierarchy.appendleft(rl_state.parent_node)
        while hierarchy[-1].parent_node is not None:
            hierarchy.append(hierarchy[-1].parent_node)
        hierarchy.appendleft(rl_state)
    return list(hierarchy)


def create_experiment_folder(main_folder_name='Experiments', inner_folder_prefix='Nero'):
    directory = os.getcwd() + '/' + main_folder_name + '/' + inner_folder_prefix
    main_folder_name = str(datetime.datetime.now())
    path_of_folder = directory + main_folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger




def selective_2Dplot(pre_trained_nero, concepts, f=sns.scatterplot, path_save_fig='2dplot.png'):
    for c in concepts:
        c.embeddings = pre_trained_nero.positive_expression_embeddings(
            c.str_individuals).cpu().detach().numpy().flatten()

    low_emb = PCA(n_components=2).fit_transform([c.embeddings for c in concepts])
    f(x=low_emb[:, 0], y=low_emb[:, 1])  # sns.scatterplot or     sns.regplot(x=low_emb[:, 0], y=low_emb[:, 1])

    for (c, emb) in zip(concepts, low_emb):
        plt.annotate(c.name, (emb[0], emb[1]), annotation_clip=False)

    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.ylim(-200, 300)
    plt.xlim(-200, 800)
    plt.savefig(path_save_fig)
    plt.plot()
    plt.show()


def validate(model, lp, args, info):
    print(f'{info}')
    model.eval()
    results = dict()

    for _, (p, n) in enumerate(lp):
        with torch.no_grad():
            ith_results = model.fit(str_pos=p, str_neg=n, topk=args['topk'])
            ith_results.update({'P': p, 'N': n,
                                'F-measure': f_measure(instances=ncel_report['Instances'],
                                                       positive_examples=set(p),
                                                       negative_examples=set(n))})

            results[_] = ith_results
    avg_f1_ncel = np.array([i['F-measure'] for i in results.values()]).mean()
    avg_runtime_ncel = np.array([i['Runtime'] for i in results.values()]).mean()
    avg_expression_ncel = np.array([i['NumClassTested'] for i in results.values()]).mean()
    print(
        f'Avg. F-measure NERO:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} LPs ')
    print('Validation Ends')


def select_loss_and_optim(loss_func_name: str, learning_rate: float, model_params) -> Tuple:
    # During our training, using MSE results in better results, less num of concepts explored
    if loss_func_name == 'MSELoss':
        loss_func = torch.nn.MSELoss()
    elif loss_func_name == 'CrossEntropyLoss':
        loss_func = torch.nn.CrossEntropyLoss()
    elif loss_func_name == 'HuberLoss':
        loss_func = torch.nn.HuberLoss()
    else:
        raise KeyError

    optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    return loss_func, optimizer

