from theano_run import main
from argparse import Namespace
import pandas as pd
from theano_run import load_higgs_data

'''
Script to run experiments using the Theano neural net.
experiments to run:
1. negative log likelihood non adversarial
2. negative log likelihood adversarial
3. cross entropy non adversarial
4. cross entropy adversarial
'''

DATA_FILE = 'training.csv'

# we'll use the same data split on all three runs.
datasets = load_higgs_data(data_file=DATA_FILE, valid_size=0.1, normalize=True)
num_classes = 2

# generic namespace that the other experiments will use.
generic_ns = Namespace(problem='higgs',
                       adv=False,
                       normalize=True,
                       n_epochs=100,
                       valid_size=0.1,
                       n_hidden=600,
                       data_file=DATA_FILE,
                       cost='neg_log')

# run experiment 1: negative log likelihood with non adversarial training.
argv_exp_1 = generic_ns
argv_exp_1.cost = 'neg_log'
argv_exp_1.adv = False

experiment_1_loss = main(argv_exp_1, datasets, num_classes)


# run experiment 2: negative log likelihood with adversarial training.
argv_exp_2 = generic_ns
argv_exp_2.cost = 'neg_log'
argv_exp_2.adv = True

experiment_2_loss = main(argv_exp_2, datasets, num_classes)

# run experiment 3: cross entropy without adversarial training
argv_exp_3 = generic_ns
argv_exp_3.cost = 'cross_ent'
argv_exp_3.adv = False

experiment_3_loss = main(argv_exp_3, datasets, num_classes)

# run experiment 4: cross entropy with adversarial training
argv_exp_4 = generic_ns
argv_exp_4.cost = 'cross_ent'
argv_exp_4.adv = True

experiment_4_loss = main(argv_exp_4, datasets, num_classes)

# print out results, using pandas.
df = pd.DataFrame({'adversarial': {'negative log likelihood': experiment_2_loss,
                                   'cross entropy': experiment_4_loss},
                   'non-adversarial': {'negative log likelihood': experiment_1_loss,
                                       'cross entropy': experiment_3_loss}})

print(df)
