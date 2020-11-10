from logging_exp import experiment_logger
import json

def log_experiment(path='../output/results_fine.json', model='baseline', level='fine', dataset='mavd' ):
    with open(path, 'r') as fp:
        results = json.load(fp)
    for k in results['class_wise']:
        logger = experiment_logger()
        logger.add_params({'model': model, 'level': level, 'dataset': dataset, 'class': k})
        for key, val in results['class_wise'][k].iteritems():
            if type(val[0]) == dict:
                for t in val:
                    for metric in val[t]:
                        logger.log_metrics(metric, results['class_wise'][k][t][metric])
            else:
                logger.log_metrics(key, val[key])
        logger.end()


log_experiment()
