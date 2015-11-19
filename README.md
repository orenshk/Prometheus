Set of machine learning algorithms for the HiggsML challenge.

# requirements:
Python version >= 3.5.0

```shell
pip3 install numpy==1.9.3
pip3 install PyYAML==3.11
pip3 install scikit-learn==0.17
pip3 install scipy==0.16.0
pip3 install sklearn==0.0
```

# how to use
## SVM

Run `python3 svms/Prometheus.py`

### optional parameters:

```shell
--email recepient1, recepient2, ..., recepientN <-- send email notification to these addresses when a fold is done.
--n_folds <num> <-- number of folds to use in a cross validation. Defaults to 25
--data_file <file_name> <-- name of file containing data. Defaults to training.csv. File must be in data folder.
```

example:

```shell
python3 svms/Prometheus.py --email my@email.com, yours@email.ca --n_folds 10 --data_file training_small.csv
```

### email requirements

Create a config.yaml file in the config folder.

```YAML
email:
  service: gmail
  username: you@gmail.com
  password: abcd1234
```