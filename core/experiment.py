from logging import Logger

from ontolearn import KnowledgeBase

from typing import List, Tuple, Set, Dict
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.binders import DLLearnerBinder

from .static_funcs import *
from .util_classes import *
from .trainer import Trainer

import numpy as np
import pandas as pd
from collections import deque
import os
from random import randint
import time


class Experiment:
    """ Main class for conducting experiments """

    def __init__(self, args):
        self.args = args

        # (1) Create Logging & Experiment folder for serialization
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Experimenter', p=self.storage_path)
        self.args['storage_path'] = self.storage_path

        # (2) Initialize KB
        self.logger.info('Knowledge Base being Initialized')
        kb = KnowledgeBase(path=self.args['path_knowledge_base'],
                           reasoner_factory=ClosedWorld_ReasonerFactory)
        self.logger.info(kb)
        self.args['num_instances'] = kb.individuals_count()
        self.args['num_named_classes'] = len([i for i in kb.ontology().classes_in_signature()])

        # (3) Initialize Learning problems
        self.logger.info('Learning Problems being generated')
        e_pos, e_neg, instance_idx_mapping, target_class_expressions = generate_training_data(kb, self.args,
                                                                                              logger=self.logger)

        self.lp = LP(e_pos=e_pos, e_neg=e_neg, instance_idx_mapping=instance_idx_mapping,
                     target_class_expressions=target_class_expressions)
        # Delete the pointers :)
        del kb, e_pos, e_neg, instance_idx_mapping, target_class_expressions
        self.logger.info(self.lp)
        # (4) Init Trainer
        self.trainer = Trainer(learning_problems=self.lp, args=self.args, logger=self.logger)
        self.logger.info(self.trainer)

        self.instance_str = list(self.lp.instance_idx_mapping.keys())
        self.__describe_and_store()

    def __describe_and_store(self):
        assert self.args['num_instances'] > 0
        # Sanity checking
        # cuda device
        self.logger.info('Device:{0}'.format(self.trainer.device))
        if torch.cuda.is_available():
            self.logger.info('Name of selected Device:{0}'.format(torch.cuda.get_device_name(self.trainer.device)))
        # (1) Store Learning Problems
        save_as_json(storage_path=self.storage_path,
                     obj={i: {'Pos': e_pos, 'Neg': e_neg} for i, (e_pos, e_neg) in
                          enumerate(zip(self.lp.e_pos, self.lp.e_neg))},
                     name='training_learning_problems')
        # (2) Store Integer mapping of instance: index of individuals
        save_as_json(storage_path=self.storage_path,
                     obj=self.lp.instance_idx_mapping, name='instance_idx_mapping')
        # (3) Store Target Class Expressions with respective expression chain from T -> ... -> TargetExp
        save_as_json(storage_path=self.storage_path, obj={target_cl.label_id: {'label_id': target_cl.label_id,
                                                                               'name': target_cl.name,
                                                                               'expression_chain': target_cl.expression_chain,
                                                                               'individuals': list(
                                                                                   target_cl.individuals),
                                                                               'idx_individuals': list(
                                                                                   target_cl.idx_individuals),
                                                                               }
                                                          for target_cl in self.lp.target_class_expressions},
                     name='target_class_expressions')

        self.args['num_outputs'] = len(self.lp.target_class_expressions)
        # (4) Store input settings
        save_as_json(storage_path=self.storage_path, obj=self.args, name='settings')
        # (5) Log details about input KB.
        self.logger.info('Describe the experiment')
        self.logger.info(
            f'Number of named classes: {self.args["num_named_classes"]}\t'
            f'Number of individuals: {self.args["num_instances"]}'
        )

    def start(self):
        # (1) Train model
        self.logger.info('Experiment starts')
        start_time = time.time()
        ncel = self.trainer.start()

        # (2) Load learning problems
        with open(self.args['path_lp']) as json_file:
            settings = json.load(json_file)
        lp = [(list(v['positive_examples']), list(v['negative_examples'])) for k, v in settings['problems'].items()]

        # (2) Evaluate model
        self.evaluate(ncel, lp, self.args)

        self.logger.info(f'Total Runtime of the experiment:{time.time() - start_time}')

    def evaluate(self, ncel, lp, args):
        self.logger.info('Evaluation Starts')

        ncel_results = dict()
        celoe_results = dict()
        # (1) Iterate over input learning problems.
        for _, (p, n) in enumerate(lp):
            ncel_report = ncel.fit(pos=p, neg=n, topK=args['topK'], local_search=False)
            ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                       positive_examples=set(p),
                                                                       negative_examples=set(n)),
                                })

            ncel_results[_] = ncel_report
            if args['eval_dl_learner']:
                celoe = DLLearnerBinder(binary_path=args['dl_learner_binary_path'], kb_path=args['path_knowledge_base'],
                                        model='celoe')
                best_pred_celoe = celoe.fit(pos=p, neg=n, max_runtime=3).best_hypothesis()
                celoe_results[_] = {'P': p, 'N': n,
                                    'Prediction': best_pred_celoe['Prediction'],
                                    'F-measure': best_pred_celoe['F-measure'],
                                    'NumClassTested': best_pred_celoe['NumClassTested'],
                                    'Runtime': best_pred_celoe['Runtime'],
                                    }
        avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
        avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
        avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()
        self.logger.info(
            f'Average F-measure NCEL:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} ')
        if args['eval_dl_learner']:
            avg_f1_celoe = np.array([i['F-measure'] for i in celoe_results.values()]).mean()
            avg_runtime_celoe = np.array([i['Runtime'] for i in celoe_results.values()]).mean()
            avg_expression_celoe = np.array([i['NumClassTested'] for i in celoe_results.values()]).mean()

            self.logger.info(
                f'Average F-measure CELOE:{avg_f1_celoe}\t Avg. Runtime:{avg_runtime_celoe}\t Avg. Expression Tested:{avg_expression_celoe}')
        self.logger.info('Evaluation Ends')


def generate_training_data(kb, args, logger):
    """

    :param logger:
    :param kb:
    :param args:
    :return:
    """
    # (1) Individual to integer mapping
    instance_idx_mapping = {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
    logger.info(f'Number of instances: {len(instance_idx_mapping)}')
    number_of_target_expressions = args['number_of_target_expressions']
    # (2) Select labels
    if args['target_expression_selection'] == 'diverse_target_expression_selection':
        target_class_expressions = diverse_target_expression_selection(kb,
                                                                       args['tolerance_for_search_unique_target_exp'],
                                                                       number_of_target_expressions,
                                                                       instance_idx_mapping,
                                                                       logger)
    elif args['target_expression_selection'] == 'random_target_expression_selection':
        target_class_expressions = random_target_expression_selection(kb,
                                                                      number_of_target_expressions,
                                                                      instance_idx_mapping,
                                                                      logger)
    else:
        raise KeyError(f'target_expression_selection:{args["target_expression_selection"]}')

    logger.info(f'Number of created target expressions: {len(target_class_expressions)}')

    (e_pos, e_neg) = generate_random_learning_problems(instance_idx_mapping, args)

    assert len(e_pos) == len(e_neg)

    logger.info(f'Number of generated learning problems : {len(e_pos)}')

    return e_pos, e_neg, instance_idx_mapping, target_class_expressions
    # return {'e_pos': e_pos, 'e_neg': e_neg, 'instance_idx_mapping': instance_idx_mapping,
    #        'target_class_expressions': target_class_expressions}


def target_expressions_via_refining_top(rho, kb, number_of_target_expressions, num_of_all_individuals,
                                        instance_idx_mapping):
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    renderer = DLSyntaxObjectRenderer()
    target_class_expressions = set()
    target_instance_set = set()
    quantifiers = set()

    # (1) Refine Top concept
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        # (3) Continue only concept is not empty.
        if num_of_all_individuals > len(i.instances) > 0:
            # (3.1) Add OWL class expression if, its instances is not already seen
            poss_target_individuals = frozenset(_.get_iri().as_str() for _ in i.instances)
            if poss_target_individuals not in target_instance_set:
                # (3.1.) Add instances
                target_instance_set.add(poss_target_individuals)
                # ( 3.2.) Create an instance
                target = TargetClassExpression(
                    label_id=len(target_instance_set),
                    name=renderer.render(i.concept),
                    individuals=poss_target_individuals,
                    idx_individuals=frozenset(
                        instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances),
                    expression_chain=[renderer.render(x.concept) for x in
                                      retrieve_concept_chain(i)]
                )
                # Add the created instance
                target_class_expressions.add(target)

            # (4) Store for later refinement if concept is \forall or \exists
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                logger.info(f'{number_of_target_expressions} target expressions generated')
                break

    return target_class_expressions, target_instance_set, quantifiers


def refine_selected_expressions(rho, kb, quantifiers, target_class_expressions, target_instance_set,
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
                    poss_target_individuals = frozenset(_.get_iri().as_str() for _ in ref_selected_states.instances)
                    if poss_target_individuals not in target_instance_set:
                        # (3.1.) Add instances
                        target_instance_set.add(poss_target_individuals)
                        # ( 3.2.) Create an instance
                        target = TargetClassExpression(
                            label_id=len(target_instance_set),
                            name=renderer.render(ref_selected_states.concept),
                            individuals=poss_target_individuals,
                            idx_individuals=frozenset(
                                instance_idx_mapping[_.get_iri().as_str()] for _ in ref_selected_states.instances),
                            expression_chain=[renderer.render(x.concept) for x in
                                              retrieve_concept_chain(ref_selected_states)]
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


def intersect_and_union_expressions_from_iterable(target_class_expressions,target_instance_set,number_of_target_expressions):
    while len(target_instance_set) < number_of_target_expressions:

        #print(f'{len(target_class_expressions)} is created as a result of refining the top concept and quantifiers.')

        res = set()
        for i in target_class_expressions:
            for j in target_class_expressions:

                if i == j:
                    continue

                i_and_j = i * j
                if len(i_and_j.individuals) > 0 and (i_and_j.individuals not in target_instance_set):
                    res.add(i_and_j)
                    target_instance_set.add(i_and_j.individuals)
                    i_and_j.label_id = len(target_instance_set)
                else:
                    del i_and_j

                if len(target_instance_set) >= number_of_target_expressions:
                    break

                i_or_j = i + j
                if len(i_or_j.individuals) > 0 and (i_or_j.individuals not in target_instance_set):
                    res.add(i_or_j)
                    target_instance_set.add(i_or_j.individuals)
                    i_or_j.label_id = len(target_instance_set)
                else:
                    del i_or_j

                if len(target_instance_set) >= number_of_target_expressions:
                    break
        target_class_expressions.update(res)


def diverse_target_expression_selection(kb, tolerance_for_search_unique_target_exp, number_of_target_expressions,
                                        instance_idx_mapping, logger) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    (1) Refine Top expression and obtain all possible ALC expressions up to length 3
    (1.1) Consider only those expression as labels whose set of individuals has not been seen before
    (1.2.) E.g. {{....}, {.}, {...}}. Only  consider those expressions as labels that do not cover all individuals
    (2)
    Select Target Expression
    :return:
    """
    # Preparation
    rho = LengthBasedRefinement(knowledge_base=kb)
    num_of_all_individuals = kb.individuals_count()
    target_class_expressions, target_instance_set, quantifiers = target_expressions_via_refining_top(rho=rho,
                                                                                                     kb=kb,
                                                                                                     number_of_target_expressions=number_of_target_expressions,
                                                                                                     num_of_all_individuals=num_of_all_individuals,
                                                                                                     instance_idx_mapping=instance_idx_mapping)
    """
    # (1) Refine Top concept
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        # (3) Continue only concept is not empty.
        if num_of_all_individuals > len(i.instances) > 0:
            # (3.1) Add OWL class expression if, its instances is not already seen
            poss_target_individuals = frozenset(_.get_iri().as_str() for _ in i.instances)
            if poss_target_individuals not in target_instance_set:
                # (3.1.) Add instances
                target_instance_set.add(poss_target_individuals)
                # ( 3.2.) Create an instance
                target = TargetClassExpression(
                    label_id=len(target_instance_set),
                    name=renderer.render(i.concept),
                    individuals=poss_target_individuals,
                    idx_individuals=frozenset(
                        instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances),
                    expression_chain=[renderer.render(x.concept) for x in
                                      retrieve_concept_chain(i)]
                )
                # Add the created instance
                target_class_expressions.add(target)

            # (4) Store for later refinement if concept is \forall or \exists
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                logger.info(f'{number_of_target_expressions} target expressions generated')
                break
    """

    logger.info(f'Refining top expression: We have {len(target_class_expressions)} number of target expressions')
    assert len(target_instance_set) == len(target_class_expressions)

    refine_selected_expressions(rho, kb, quantifiers, target_class_expressions, target_instance_set,
                                tolerance_for_search_unique_target_exp, instance_idx_mapping,
                                number_of_target_expressions, num_of_all_individuals)
    logger.info(
        f'Refining top expression + quantifiers : We have {len(target_class_expressions)} number of target expressions')
    assert len(target_instance_set) == len(target_class_expressions)
    """
    # (5) Refine
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
                    poss_target_individuals = frozenset(_.get_iri().as_str() for _ in ref_selected_states.instances)
                    if poss_target_individuals not in target_instance_set:
                        # (3.1.) Add instances
                        target_instance_set.add(poss_target_individuals)
                        # ( 3.2.) Create an instance
                        target = TargetClassExpression(
                            label_id=len(target_instance_set),
                            name=renderer.render(ref_selected_states.concept),
                            individuals=poss_target_individuals,
                            idx_individuals=frozenset(
                                instance_idx_mapping[_.get_iri().as_str()] for _ in ref_selected_states.instances),
                            expression_chain=[renderer.render(x.concept) for x in
                                              retrieve_concept_chain(ref_selected_states)]
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
    assert len(target_instance_set) == len(target_class_expressions)
    logger.info(
        f'Refining top expression + quantifiers : We have {len(target_class_expressions)} number of target expressions')
    """

    intersect_and_union_expressions_from_iterable(target_class_expressions, target_instance_set,number_of_target_expressions)
    logger.info(
        f'Refining top expression + quantifiers : We have {len(target_class_expressions)} number of target expressions')
    """
    while len(target_instance_set) < number_of_target_expressions:

        logger.info(
            f'{len(target_class_expressions)} is created as a result of refining the top concept and quantifiers.')

        res = set()
        for i in target_class_expressions:
            for j in target_class_expressions:

                if i == j:
                    continue

                i_and_j = i * j
                if len(i_and_j.individuals) > 0 and (i_and_j.individuals not in target_instance_set):
                    res.add(i_and_j)
                    target_instance_set.add(i_and_j.individuals)
                    i_and_j.label_id = len(target_instance_set)
                else:
                    del i_and_j

                if len(target_instance_set) >= number_of_target_expressions:
                    break

                i_or_j = i + j
                if len(i_or_j.individuals) > 0 and (i_or_j.individuals not in target_instance_set):
                    res.add(i_or_j)
                    target_instance_set.add(i_or_j.individuals)
                    i_or_j.label_id = len(target_instance_set)
                else:
                    del i_or_j

                if len(target_instance_set) >= number_of_target_expressions:
                    break
        target_class_expressions.update(res)
    """
    result = []
    for ith, tce in enumerate(target_class_expressions):
        tce.label_id = ith
        result.append(tce)
    return result


def random_target_expression_selection(kb, number_of_target_expressions, instance_idx_mapping, logger) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    Select Target Expression
    :return:
    """
    # @TODO followed same method of not using RL_State as done in entropy_based_target_expression_selection
    # (1) Preparation
    renderer = DLSyntaxObjectRenderer()
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
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
                logger.info(f'{number_of_target_expressions} target expressions generated')
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
            individuals=frozenset(_.get_iri().as_str() for _ in i.instances),
            idx_individuals=frozenset(instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances),
            expression_chain=[renderer.render(x.concept) for x in retrieve_concept_chain(i)]
        )
        labels.append(target)
    return labels


def generate_random_learning_problems(instance_idx_mapping: Dict,
                                      args: Dict) -> Tuple[List[int], List[int]]:
    """
    Generate Learning problems
    :param instance_idx_mapping:
    :param target_idx_individuals:
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

    return pos_examples, neg_examples
