import warnings
warnings.filterwarnings("ignore")

import configargparse
import yaml

import autosklearn.classification
import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics
import pandas as pd

from joblib import dump, load
import json


autosklearn_metrics = {'accuracy': autosklearn.metrics.accuracy,
                   'balanced_accuracy': autosklearn.metrics.balanced_accuracy,
                   'f1': autosklearn.metrics.f1,
                   'f1_macro': autosklearn.metrics.f1_macro,
                   'f1_micro': autosklearn.metrics.f1_micro,
                   'f1_samples': autosklearn.metrics.f1_samples,
                   'f1_weighted': autosklearn.metrics.f1_weighted,
                   'roc_auc': autosklearn.metrics.roc_auc,
                   'precision': autosklearn.metrics.precision,
                   'precision_macro': autosklearn.metrics.precision_macro,
                   'precision_micro': autosklearn.metrics.precision_micro,
                   'precision_samples': autosklearn.metrics.precision_samples,
                   'precision_weighted': autosklearn.metrics.precision_weighted,
                   'average_precision': autosklearn.metrics.average_precision,
                   'recall': autosklearn.metrics.recall,
                   'recall_macro': autosklearn.metrics.recall_macro,
                   'recall_micro': autosklearn.metrics.recall_micro,
                   'recall_samples': autosklearn.metrics.recall_samples,
                   'recall_weighted': autosklearn.metrics.recall_weighted,
                   'log_loss': autosklearn.metrics.log_loss,
                   'r2': autosklearn.metrics.r2,
                   'mean_squared_error': autosklearn.metrics.mean_squared_error,
                   'mean_absolute_error': autosklearn.metrics.mean_absolute_error,
                   'median_absolute_error': autosklearn.metrics.median_absolute_error
                  }

def load_csv(filename, header, sep):
    
    header = 0 if header else None
    
    separators = {"comma":",", "tab":r"\t", "space":r"\s", "white_spaces":r"\s+","colon":";"}

    df = pd.read_csv(filename, sep=separators.get(sep), header=header, error_bad_lines=False)

    verboseprint("Loaded file %s with dimensions %s" % (filename, df.shape))
    verboseprint("Data types:\n")
    verboseprint(df.dtypes)
    verboseprint(df.head())
    
    return df

def run_autosklearn(filename, target, task, header, sep, holdout, seed, time_for_task, per_run_time_limit, ensemble_size, ensemble_nbest, max_models_on_disc, memory_limit, include, exclude, resampling_strategy, resampling_strategy_arguments, metric, scoring_functions):
    
    print(resampling_strategy_arguments)
    
    df = load_csv(filename, header, sep)
        
    X = df.drop(df[[target]], axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=seed, test_size=holdout)
    
    if per_run_time_limit == None:
        per_run_time_limit = int(time_for_task / 10)

    scorer = autosklearn_metrics[metric]
    
    for idx, s in enumerate(scoring_functions):
        scoring_functions[idx] = autosklearn_metrics[s[0]]        

    task = task.lower()
        
    if task=='classification':

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_for_task,
            per_run_time_limit=per_run_time_limit,
            ensemble_size=ensemble_size,
            ensemble_nbest = ensemble_nbest,
            max_models_on_disc = max_models_on_disc,
            include = include,
            exclude=exclude,
            resampling_strategy= resampling_strategy,
            resampling_strategy_arguments = resampling_strategy_arguments,
            seed = seed,
            metric=scorer,
            scoring_functions = scoring_functions,
            initial_configurations_via_metalearning=0,
            n_jobs=-1
        )
    
    elif task == 'regression':
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=time_for_task,
            per_run_time_limit=per_run_time_limit,
            ensemble_size=ensemble_size,
            ensemble_nbest = ensemble_nbest,
            max_models_on_disc = max_models_on_disc,
            include = include,
            exclude=exclude,
            resampling_strategy= resampling_strategy,
            resampling_strategy_arguments = resampling_strategy_arguments,
            seed = seed,
            metric=scorer,
            scoring_functions = scoring_functions,
            n_jobs=-1
            )

    automl.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    #save the predicitons
    predictions = automl.predict(X_test)
    
    if task == 'classification':
        metric_out = sklearn.metrics.roc_auc_score(y_test, predictions)
        metric_out_name ='roc_auc'
        output_col_name = 'param_classifier:__choice__'
    elif task == 'regression':
        metric_out = sklearn.metrics.mean_squared_error(y_test, predictions)
        metric_out_name ='Mean Squared Error'
        output_col_name = 'param_regressor:__choice__'
    
    print('Sprint Stats:')
    print(automl.sprint_statistics())
    print(' ')
    print('-----------------------------------------')
    print(' ')
    print('Holdout score (' + metric_out_name + '):')
    print(metric_out)
    
    dump(automl, '/mnt/artifacts/automl.joblib')

    with open('/mnt/artifacts/dominostats.json', 'w') as f:
        f.write(json.dumps( {metric_out_name : metric_out}))
        
    automl_results = pd.DataFrame.from_dict(automl.cv_results_, orient='columns')
    
    cols_to_print = ['rank_test_scores', 'mean_test_score', output_col_name]
    user_metrics = [c for c in automl_results.columns if 'metric_' in c]
    cols_to_print.extend(user_metrics)
        
    verboseprint(automl_results[cols_to_print].sort_values(by='rank_test_scores', ascending=True).head(20))
    
    automl_results.to_csv('/mnt/artifacts/automl_results.csv', index=False)
    

if __name__ == "__main__":
    
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

        if x <= 0.0 or x >= 1.0:
            raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0)"%(x,))
        return x
    
    def list_str(values):
        return values.split(",")
        
    parser = configargparse.ArgumentParser(
        description = "AutoML script using auto-sklearn, from Domino Data Lab",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        )
    
    parser.add(
        "--config",
        is_config_file=True, 
        help="config file path"
        )
    
    parser.add_argument("--file",
                        type=str,
                        required=False,
                        help="Path to training data", 
                        default="./data/raw/heart.csv")
    
    parser.add_argument("--target",
                        type=str,
                        required=True, 
                        help="Target variable column name")
    
    parser.add_argument("--task",
                        type=str,
                        required=True,
                        help="Model type (regression or classifcation)",
                        choices=["classification", "regression"])
    
    parser.add_argument("--header",
                        type=bool,
                        required=False, 
                        help="Set to True if the first row contains a header",
                        default=False)
    
    parser.add_argument("--sep",
                        type=str,
                        required=False,
                        help="Delimiter to use",
                        default="comma",
                        choices=["comma", "tab", "space", "white_spaces","colon"])
    
    parser.add_argument("--holdout", 
                        type=restricted_float,
                        required=False,
                        help="Size of the holdout set (0,1)",
                        default=0.2)
    
    parser.add_argument("--seed",
                        type=int,
                        required=False,
                        help="Random seed used for sampling and fitting",
                        default=42)
    

    parser.add_argument("--time_left_for_this_task",
                        type=int,
                        required=False, 
                        help="Auto-sklearn: Time limit in seconds for the search of appropriate models. By increasing this value, auto-sklearn has a higher chance of finding better models.",
                        default=3600)
    
    parser.add_argument("--per_run_time_limit",
                        type=int,
                        required=False, 
                        help="Auto-sklearn: Time limit for a single call to the machine learning model. Model fitting will be terminated if the machine learning algorithm runs over the time limit. Set this value high enough so that typical machine learning algorithms can be fit on the training data.",
                        default=None)
    
    parser.add_argument("--ensemble_size",
                        type=int,
                        required=False, 
                        help="Auto-sklearn: Number of models added to the ensemble built by Ensemble selection from libraries of models. Models are drawn with replacement. If set to 0 no ensemble is fit.",
                        default=50)

    parser.add_argument("--ensemble_nbest",
                        type=int,
                        required=False, 
                        help="Auto-sklearn: Only consider the ensemble_nbest models when building an ensemble.",
                        default=50)

    parser.add_argument("--max_models_on_disc",
                        type=int,
                        required=False, 
                        help="Auto-sklearn: Defines the maximum number of models that are kept in the disc. The additional number of models are permanently deleted. Due to the nature of this variable, it sets the upper limit on how many models can be used for an ensemble. It must be an integer greater or equal than 1. If set to None, all models are kept on the disc.",
                        default=50)
    
    parser.add_argument("--memory_limit",
                        type=int,
                        required=False, 
                        help="Auto-sklearn: Memory limit in MB for the machine learning algorithm. auto-sklearn will stop fitting the machine learning algorithm if it tries to allocate more than memory_limit MB.",
                        default=3072)
    
    parser.add_argument("--include",
                        type=str,
                        required=False, 
                        help="Auto-sklearn: If None, all possible algorithms are used. Otherwise, specifies a step and the components that are included in search",
                        default=None)
    
    parser.add_argument("--exclude",
                        type=str,
                        required=False, 
                        help="Auto-sklearn: If None, all possible algorithms are used. Otherwise, specifies a step and the components that are excluded from search",
                        default=None)
    
    parser.add_argument("--resampling_strategy",
                        type=str,
                        required=False, 
                        help="Auto-sklearn: Internal partitioning scheme used during model tuning & fit",
                        default='holdout',
                        choices=["holdout", "holdout-iterative-fit", "cv", "cv-iterative-fit","partial-cv"])
        
    parser.add_argument("--resampling_strategy_arguments",
                        type=str,
                        required=False, 
                        help="Auto-sklearn: Additional arguments for resampling_strategy",
                        default=None,)
    
    parser.add_argument("--metric",
                        type=str,
                        required=False, 
                        help="Auto-sklearn: Metric uised for paramaater tuning and model ranking. An instance of autosklearn.metrics.Scorer",
                        default=None)
    
    parser.add_argument("--scoring_functions",
                        type=list_str,
                        action="append",
                        required=False, 
                        help="Auto-sklearn: List of scorers which will be calculated for each pipeline and results will be available via cv_results",
                        default=None)
    
#     parser.add_argument("--output_dir",
#                         type=str,
#                         required=False, 
#                         help="Default output directory",
#                         default="./results/myrun")
    
#     parser.add_argument("--tmp_dir",
#                         type=str,
#                         required=False,
#                         help="Temporary directory",
#                         default="tmp")
    
    parser.add_argument("--verbose", type=bool, required=False, help="Output additional information", default=True)

    args = parser.parse_args()
        
    # Parse dictionary inputs from config
    if args.include:
        include = yaml.safe_load(args.include)
    else:
        include = None
    
    if args.exclude:
        exclude = yaml.safe_load(args.exclude)
    else:
        exclude = None
        
    if args.resampling_strategy_arguments:
        resampling_strategy_arguments = yaml.safe_load(args.resampling_strategy_arguments)
    else:
        resampling_strategy_arguments = None
    
    # output_dir = args.output_dir
    # tmp_dir = os.getcwd() + os.sep + args.tmp_dir
    
    verboseprint = print if args.verbose else lambda *a, **k: None
        
    run_autosklearn(filename = args.file,
                    target = args.target,
                    task = args.task,
                    header = args.header,
                    sep = args.sep,
                    holdout = args.holdout,
                    seed = args.seed, 
                    time_for_task = args.time_left_for_this_task,
                    per_run_time_limit = args.per_run_time_limit,
                    ensemble_size = args.ensemble_size,
                    ensemble_nbest = args.ensemble_nbest,
                    max_models_on_disc = args.max_models_on_disc,
                    memory_limit = args.memory_limit,
                    include = include,
                    exclude = exclude,
                    resampling_strategy = args.resampling_strategy,
                    resampling_strategy_arguments = resampling_strategy_arguments,
                    metric = args.metric,
                    scoring_functions = args.scoring_functions,
                    # output_dir = output_dir,
                    # tmp_dir = tmp_dir
                   )