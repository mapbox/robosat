import random


class ReservoirSampler:
    '''Randomly samples k items from a stream of unknown n items.
    '''

    def __init__(self, capacity):
        '''Creates an new `ReservoirSampler` instance.

        Args:
          capacity: the number of items to randomly sample from a stream of unknown size.
        '''

        assert capacity > 0

        self.capacity = capacity
        self.reservoir = []
        self.pushed = 0

    def push(self, v):
        '''Adds an item to the reservoir.

        Args:
          v: the item from the stream to add to the reservoir.
        '''

        size = len(self.reservoir)

        if size < self.capacity:
            self.reservoir.append(v)
        else:
            assert size == self.capacity
            assert size <= self.pushed

            p = self.capacity / self.pushed

            if random.random() < p:
                i = random.randint(0, size - 1)
                self.reservoir[i] = v

        self.pushed += 1

    def __len__(self):
        '''Returns the number of randomly sampled items.

        Returns:
          The number of randomly sampled items in the reservoir.
        '''

        return len(self.reservoir)

    def __getitem__(self, k):
        '''Returns a randomly sampled item in the reservoir.

        Args:
          k: the index for the kth item from the reservoir to return.

        Returns:
          The kth item in the reservoir of randomly sampled items.
        '''

        return self.reservoir[k]

    def __repr__(self):
        '''Returns the representation for this class.

        Returns:
          The string representation for this class.
        '''

        return '<{}: {}>'.format(self.__class__.__name__, list(self))
