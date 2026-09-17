from typing import Iterator, List
from torch.utils.data.sampler import BatchSampler, Sampler, RandomSampler
import random
import numpy as np

class BalancedSampler(Sampler):
    def __init__(self, data_source = None, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.occurence_list = data_source.get_occurence_list()
        num_occurences = [len(s) for s in self.occurence_list]
        non_zero_classes = [i for i, c in enumerate(num_occurences) if c > 0]
        self.non_zero_classes = non_zero_classes

        self.num_samples = num_samples or len(self.data_source)

        self.rng = np.random.default_rng()
        probabilities = np.array([1/c for c in num_occurences if c > 0])**0.2
        probabilities /= probabilities.sum()
        
        self.probabilities = probabilities
        print(f"balanced sampler will use these probabilities: {self.probabilities}")
        


    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator:
        for _ in range(self.num_samples):
            _class = self.rng.choice(self.non_zero_classes, 1, p=self.probabilities)[0]
            sample = self.rng.choice(self.occurence_list[_class], 1)[0]
            yield sample


class BalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        base_sampler,
        batch_size: int,
        num_batches=None,
        force_classes: List[int] | None = None,
    ):
        self.base_sampler = base_sampler
        self.batch_size = batch_size
        self.occurence_list = base_sampler.occurence_list
        self.rng = base_sampler.rng

        self.num_batches = num_batches or len(base_sampler) // batch_size
        self.force_classes = force_classes or []

        if len(self.force_classes) > batch_size:
            raise ValueError("More forced classes than batch size")

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch = []

            # 1️⃣ Force one sample per requested class
            for cls in self.force_classes:
                idx = self.rng.choice(self.occurence_list[cls])
                batch.append(int(idx))

            # 2️⃣ Fill remainder using original sampler
            remaining = self.batch_size - len(batch)
            base_iter = iter(self.base_sampler)

            for _ in range(remaining):
                batch.append(int(next(base_iter)))

            yield batch