from typing import Any, Dict, List, Union
import numpy as np
import logging
import os

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class RandomSearch:

    @staticmethod
    def random_choice(args: List[Any], n: int = 1):
        """
        pick a random element from a set.
        
        Example:
            >> sampler = RandomSearch.random_choice(1,2,3)
            >> sampler()
                2
        """
        choices = []
        for arg in args:
            choices.append(arg)
        if n == 1:
            return lambda: np.random.choice(choices, replace=False)
        else:
            return lambda: np.random.choice(choices, n, replace=False)

    @staticmethod
    def random_integer(low: Union[int, float], high: Union[int, float]):
        """
        pick a random integer between two bounds
        
        Example:
            >> sampler = RandomSearch.random_integer(1, 10)
            >> sampler()
                9
        """
        return lambda: int(np.random.randint(low, high))

    @staticmethod
    def random_loguniform(low: Union[float, int], high: Union[float, int]):
        """
        pick a random float between two bounds, using loguniform distribution
        
        Example:
            >> sampler = RandomSearch.random_loguniform(1e-5, 1e-2)
            >> sampler()
                0.0004
        """
        return lambda: np.exp(np.random.uniform(np.log(low), np.log(high)))

    @staticmethod
    def random_uniform(low: Union[float, int], high: Union[float, int]):
        """
        pick a random float between two bounds, using uniform distribution
        
        Example:
            >> sampler = RandomSearch.random_uniform(0, 1)
            >> sampler()
                0.01
        """
        return lambda: np.random.uniform(low, high)


class HyperparameterSearch:

    def __init__(self, **kwargs):
        self.search_space = {}
        self.lambda_ = lambda: 0
        for key, val in kwargs.items():
            self.search_space[key] = val

    def parse(self, val: Any):
            
        if isinstance(val, (int, np.int)):
            return int(val)
        elif isinstance(val, (float, np.float)):
            return val
        elif isinstance(val, (np.ndarray, list)):
            return " ".join(val)
        elif val is None:
            return None
        if isinstance(val, str):
            return val
        else:
            val = val()
            if isinstance(val, (int, np.int)):
                return int(val)
            elif isinstance(val, (np.ndarray, list)):
                return " ".join(val)
            else:
                return val


    def sample(self) -> Dict:
        res = {}
        for key, val in self.search_space.items():
            try:
                res[key] = self.parse(val)
            except (TypeError, ValueError) as error:
                logger.error(f"Could not parse key {key} with value {val}. {error}")

        return res

    def update_environment(self, sample) -> None:
        for key, val in sample.items():
            os.environ[key] = str(val)


SEARCH_SPACE = {
        "penalty": RandomSearch.random_choice(["l1", "l2"]),
        "C": RandomSearch.random_uniform(0, 1),
        "solver": "liblinear",
        "multi_class": "auto",
        "tol": RandomSearch.random_loguniform(10e-5, 10e-3),
        "stopwords": RandomSearch.random_choice([0, 1]),
        "weight": RandomSearch.random_choice(["hash"]),
        "ngram_range": RandomSearch.random_choice(["1 2", "2 3", "1 3"]),
        "random_state": RandomSearch.random_integer(0, 100000)
}
BEST_HPS = {
        "penalty": "l1",
        "C": 0.977778,
        "multi_class": "auto",
        "solver": "liblinear",
        "tol": 0.000816,
        "ngram_range": "1 2",
        "random_state": 44555,
        "weight": "hash",
        "stopwords": None
}
