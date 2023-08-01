"""
Deploy our approach
"""
from typing import Dict

import pandas as pd
import torch
import json
import gradio as gr
from core import NERO, DeepSet, ST, TargetClassExpression, f_measure
from random import randint
from argparse import ArgumentParser
import random

from core.loaders import *


def load_target_class_expressions_and_instance_idx_mapping(args):
    """

    :param args:
    :return:
    """
    # target_class_expressions Must be empty and must be filled in an exactorder
    target_class_expressions = []
    df = ddf.read_csv(args['path_of_experiment_folder'] + '/target_class_expressions.csv', dtype={'label_id': 'int',
                                                                                                  'name': 'object',
                                                                                                  'str_individuals': 'object',
                                                                                                  'idx_individuals': 'object',
                                                                                                  'atomic_expression': 'object',
                                                                                                  'concepts': 'object',
                                                                                                  'filler': 'object',
                                                                                                  'role': 'object',
                                                                                                  })
    df = df.compute(scheduler='processes').set_index('Unnamed: 0')

    print(df.head())

    with open(args['path_of_experiments'] + '/target_class_expressions.json', 'r') as f:
        for k, v in json.load(f).items():
            k: str  # k denotes k.th label of target expression, json loading type conversion from int to str appreantly
            v: dict  # v contains info for Target Class Expression Object
            assert isinstance(k, str)
            assert isinstance(v, dict)
            try:
                k = int(k)
            except ValueError:
                print(k)
                print('Tried to convert to int')
                exit(1)
            try:

                assert k == v['label_id']
            except AssertionError:
                print(k)
                print(v['label_id'])
                exit(1)

            t = TargetClassExpression(label_id=v['label_id'],
                                      name=v['name'],
                                      idx_individuals=frozenset(v['idx_individuals']),
                                      expression_chain=v['expression_chain'])
            assert len(t.idx_individuals) == len(v['idx_individuals'])

            target_class_expressions.append(t)

    instance_idx_mapping = dict()
    with open(args['path_of_experiments'] + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))

    return target_class_expressions, instance_idx_mapping


def load_pytorch_module(args: Dict) -> torch.nn.Module:
    """ Load weights and initialize pytorch module"""
    # (1) Load weights from experiment repo
    weights = torch.load(args['path_of_experiments'] + '/final_model.pt', torch.device('cpu'))
    if args['neural_architecture'] == 'DeepSet':
        model = DeepSet(args)
    elif args['neural_architecture'] == 'ST':
        model = ST(args)
    else:
        raise NotImplementedError('There is no other model')
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def load_ncel(args: Dict) -> NERO:
    # (2) Load target class expressions & instance_idx_mapping
    target_class_expressions, instance_idx_mapping = load_target_class_expressions_and_instance_idx_mapping(args)
    # (1) Load Pytorch Module
    model = load_pytorch_module(args)

    model = NERO(model=model,
                 quality_func=f_measure,
                 target_class_expressions=target_class_expressions,
                 instance_idx_mapping=instance_idx_mapping)
    model.eval()
    return model


def launch_service(nero):
    def predict(positive_examples, negative_examples, size_of_examples, random_examples: bool):
        if random_examples:
            # Either sample from here self.instance_idx_mapping
            # or sample from targets
            pos_str = random.sample(list(nero.instance_idx_mapping.keys()), int(size_of_examples))
            neg_str = random.sample(list(nero.instance_idx_mapping.keys()), int(size_of_examples))
        else:
            pos_str = positive_examples.split(",")
            neg_str = negative_examples.split(",")

        with torch.no_grad():
            report = nero.fit(str_pos=pos_str, str_neg=neg_str, topk=100)
        if len(pos_str) < 20:
            s = f'E^+:{",".join(pos_str)}\nE^-:{",".join(neg_str)}\n'
        else:
            s = f'|E^+|:{len(pos_str)}\n|E^-|:{len(neg_str)}\n'

        report.pop('Instances')
        return s, pd.DataFrame([report])

    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label='Positive Examples'),
                gr.inputs.Textbox(lines=5, placeholder=None, label='Negative Examples'),
                gr.inputs.Slider(minimum=1, maximum=100),
                "checkbox"],
        outputs=[gr.outputs.Textbox(label='Learning Problem'), gr.outputs.Dataframe(label='Predictions',type='pandas')],
        title='Rapid Induction of Description Logic Expressions via Nero',
        description='Click Random Examples & Submit.').launch(server_name="0.0.0.0",server_port=7860,share=False)


def run(args):
    print('Loading Nero...')
    ncel_model, loading_time_to_add = load_nero(args)
    print(f'Nero is loaded:{loading_time_to_add}')
    launch_service(ncel_model)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_of_experiment_folder", type=str, default='Best/NeroFamily')

    # Inference Related
    parser.add_argument("--topK", type=int, default=100,
                        help='Test the highest topK target expressions')
    parser.add_argument("--use_multiprocessing_at_parsing", type=int,
                        default=0, help='1 or 0')
    parser.add_argument('--use_search', default='None', help='None,SmartInit')
    run(parser.parse_args())
