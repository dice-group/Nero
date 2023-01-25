from argparse import ArgumentParser
from core.experiment import Experiment
import torch
import random

def argparse_default(description=None):
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/Family.owl',
                        # default='KGs/DBpedia/DBpedia.owl',
                        help='The absolute path of a knowledge base required.')
    parser.add_argument("--path_lp", type=str,
                        default='LPs/Family/lp_dl_learner.json',
                        help='If None, examples are randomly generated')
    parser.add_argument("--dl_learner_binary_path", type=str, default='dllearner-1.4.0/')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of cpus used during batching')

    # Class Expression Learning
    parser.add_argument("--num_individual_per_example", type=int, default=10,
                        help='Input set size |E^+| and |E^-|.')
    parser.add_argument("--num_of_learning_problems_training", type=int, default=1,
                        help='Total number of LP => this val x |target exp|.')
    # Neural related
    parser.add_argument("--neural_architecture", type=str,
                        default='DeepSet',
                        help='Nero is based on two deep-set neural network. Later we will add set transformer')
    parser.add_argument("--quality_function_training", type=str,
                        default='fmeasure', )
    parser.add_argument("--loss_func", type=str,
                        default='MSELoss',
                        help='[MSELoss,HuberLoss,CrossEntropyLoss]')
    parser.add_argument("--number_of_target_expressions", type=int,
                        default=1000)
    parser.add_argument("--target_expression_selection", type=str,
                        default='uncorrelated_target_expression_selection')
    # Hyperparameters of Neural Class Expression
    parser.add_argument("--num_embedding_dim", type=int, default=100, help='Number of embedding dimensions.')
    # Training Related
    parser.add_argument("--learning_rate", type=float, default=0.1, help='Learning Rate')
    parser.add_argument("--num_epochs", type=int, default=50, help='Number of iterations over the entire dataset.')
    parser.add_argument("--val_at_every_epochs", type=int, default=500, help='How often eval.')
    parser.add_argument("--batch_size", type=int, default=1024)
    # Inference Related
    parser.add_argument("--topK", type=int, default=10,
                        help='Test the highest topK target expressions')
    # Analysis Related
    parser.add_argument("--plot_embeddings", type=int, default=0, help='1 => Yes, 0 => No')
    parser.add_argument('--use_search', default='None', help='None,SmartInit')
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)

def main(args):
    Experiment(args).start()


# OWL version 1 and OWL 2, our library only works with OWL 2
if __name__ == '__main__':
    main(vars(argparse_default()))
