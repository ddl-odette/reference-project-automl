import sys
import auto_sklearn

file = sys.argv[1]
target = sys.argv[2]
task = sys.argv[3]

auto_sklearn.run_autosklearn(filename=file,
                  target=target,
                  task=task,
                  header=1,
                  sep='comma',
                  holdout=0.2,
                  seed=42,
                  time_for_task=3600,
                  per_run_time_limit=None,
                  ensemble_size=50,
                  ensemble_nbest=50,
                  max_models_on_disc=50,
                  memory_limit=3072,
                  include=None,
                  exclude=None,
                  resampling_strategy='cv',
                  resampling_strategy_arguments= {'train_size': 0.8,
                                                  'shuffle': True,
                                                  'folds': 5},
                  metric=None,
                  scoring_functions=None
                            )