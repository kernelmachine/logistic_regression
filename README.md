# Logistic Regression Classification Baseline


This repository provides a simple logistic regression classification baseline for NLP research in text classification.

Through simple commands, one can:

    * Run random search trials over a variety of LR hyperparameters, including those involving the input representation.
    * Run cross-validation/jackknifing if dev set is not available
    * Run experiments with (possibly stratified) subsamples of the training data
    * Parallelize experiments using `gnu parallel`
    * Visualize the effect of individual hyperparameters on classification performance

This repository just expects a `train.jsonl` file, in JSON lines format, each line corresponding to the format `{"text":..., "label":...}`. You can also supply a `dev.jsonl` file. If you don't, we will jackknife the training data and report performance metrics over all splits. 

## Run single experiment

```
python -m lr.train --train_file data/train.jsonl --dev_file data/dev.jsonl --search_trials 5 --serialization_dir model_logs/lr -o
```

## Run single experiment on (stratified) sampled data

```
python -m lr.train --train_file data/train.jsonl --dev_file data/dev.jsonl --search_trials 10 --serialization_dir model_logs/sampled_lr --train_subsample 1000 --stratified -o
```

## Run single jackknifing experiment

```
python -m lr.train --train_file data/train.jsonl --search_trials 10 --jackknife_partitions 3 --save_jackknife_partitions --serialization_dir model_logs/jackknife_lr  --stratified --train_subsample 1000 -o
```

## Evaluate on test data

```
parallel --ungroup python -m lr.train --train_file data/train.jsonl --dev_file data/dev.jsonl --test-fiele data/test.jsonl --search_trials 1  --serialization_dir model_logs/ag_lr/exp_{#} --evaluate-on-test -o ::: {1..6}
```


## Run many experiments in parallel

```
parallel --ungroup python -m lr.train --train_file data/train.jsonl --dev_file data/dev.jsonl --search_trials 1  --serialization_dir model_logs/parallel_lr/exp_{#}  -o ::: {1..6}
```


## Merge multiple experiment results

```
python -m lr.merge --experiments model_logs/parallel_lr/* --output-file model_logs/parallel_lr/master_results.jsonl
```

## Visualize scatterplot of hyperparameter vs performance

```
python -m lr.plot --hyperparameter C  --results_file parallel_lr/master_results.jsonl -p dev_f1 
```


## Visualize boxplot of hyperparameter vs performance

```
python -m lr.plot --hyperparameter weight --boxplot --results_file parallel_lr/master_results.jsonl -p dev_f1 
```

