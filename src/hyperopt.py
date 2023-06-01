from typing import Union

from hyperopt import Trials, fmin
from src.cross_validation import kfold_train


def optimize_hyperparameters(
    dataset, space, algo, max_evals, **custom_kwargs
) -> Union[dict, Trials]:
    """Use hyperopt to optimize the hyperparameters of the model.

    Args:
        dataset: dataset to use for k-fold cross-validation.
        space: hyperopt parameter space to explore.
        algo: one of hyperopt.tpe.suggest pr hyperopt.rand.suggest.
        max_evals: number of evaluation to do in the search space.

    Returns:
        Union[dict, Trials]: dictionary of best hyperparameters and the trials object.
    """
    trials = Trials()

    def kfold_train_wrapper(hyperopt_kwargs):
        return kfold_train(
            dataset=dataset,
            **custom_kwargs,
            **hyperopt_kwargs,
        )

    best = fmin(
        kfold_train_wrapper,
        space=space,
        algo=algo,
        max_evals=max_evals,
        trials=trials,
        return_argmin=False,
    )

    return best, trials
