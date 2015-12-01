import numpy as np
import sys
import argparse
import pickle

import time
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from utils.notification import email, push_note
from utils import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Prometheus svm')
    parser.add_argument('--email', nargs='*', default='')
    parser.add_argument('--test_size', type=float, default=0.5)
    parser.add_argument('--data_file', default='training.csv')
    parser.add_argument('--gridsearch', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--max_iter', default=-1, type=float)
    parser.add_argument('--cache_size', default=2500, type=int)

    args = parser.parse_args()

    all_data, all_labels = load_data(args.data_file)

    # we're going to use grid search to find best parameters.
    train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=args.test_size)

    # separate out the weights.
    train_weights = train_data[:, -1]
    train_data = train_data[:, 0:-1]

    # no need to save the test weights. Right?
    test_data = test_data[:, 0:-1]

    if args.gridsearch:
        parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-5, -3.2), 'C': [100, 1e3, 1e4, 1e5]},
                      # {'kernel': ['sigmoid'], 'gamma': np.logspace(-3, 3), 'C': [1, 10, 100, 1e3], 'coef0': [0, 1]}
                      ]

        classifier = GridSearchCV(svm.SVC(cache_size=args.cache_size),
                                  parameters,
                                  cv=3,
                                  fit_params={'sample_weight': train_weights.flat},
                                  verbose=True)

        print('starting search. This may be a while...')
        classifier.fit(train_data, train_labels.flat)

        # best parameters
        subject = 'best parameters: {} with score: {}'.format(classifier.best_params_, classifier.best_score_)
        best_params = classifier.best_params_
        param_str = '_'.join(['{}={}'.format(x, best_params[x]) for x in sorted(best_params.keys())])
        best_classifier = classifier.best_estimator_
    else:
        # we're going to use hardcoded parameters discovered through previous grid search.
        opt_C = 1e3
        opt_gamma = 0.00024883895413685354
        classifier = svm.SVC(C=opt_C,
                             kernel='rbf',
                             gamma=opt_gamma,
                             cache_size=args.cache_size,
                             max_iter=int(args.max_iter))
        print('training svm')
        sys.stdout.flush()
        start_time = time.time()
        classifier.fit(train_data, train_labels.flat, sample_weight=train_weights.flat)
        elapsed = time.time() - start_time
        subject = ('Done training SVM in {:.3f} minutes on {} samples '
                   'with parameters C={} and gamma={}'.format(elapsed / 60.0, train_data.shape[0],
                                                              opt_C,
                                                              opt_gamma))
        # it's possible that no test vector was created, if user chose test_size to be 0.0
        if test_data.any():
            subject += ' score: {}'.format(classifier.score(test_data, test_labels.flat))
        param_str = 'C={}_gamma={}_kernel=rbf'.format(opt_C, opt_gamma)
        best_classifier = classifier

    print(subject)
    if test_data.any():
        print('classification report')
        report = classification_report(y_true=test_labels, y_pred=classifier.predict(test_data))
        print(report)
    else:
        report = ''

    if args.email:
        body = report
        email(to_addrs=args.email, subject=subject, body=report)

    if args.push:
        push_note(subject=subject, body=report)

    # save the classifier.
    with open('classifier-{}.dat'.format(param_str), 'wb') as f:
        pickle.dump(best_classifier, f, protocol=3)
