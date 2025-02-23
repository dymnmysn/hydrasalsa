class KittiWaymoTrainLoader:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1.trainloader
        self.dataloader2 = dataloader2.trainloader
        self.iter1 = iter(self.dataloader1)
        self.iter2 = iter(self.dataloader2)

        # Determine the finite length based on the shorter dataloader
        self.length = min(len(dataloader1), len(dataloader2))
        self.steps = 0  # To keep track of steps within an epoch

    def __iter__(self):
        self.steps = 0  # Reset steps each time we start a new epoch
        return self

    def __next__(self):
        if self.steps >= self.length:
            raise StopIteration  # End the iteration when the epoch length is reached

        try:
            batch_data1 = next(self.iter1)
        except StopIteration:
            # Reset iterator if exhausted
            self.iter1 = iter(self.dataloader1)
            batch_data1 = next(self.iter1)

        try:
            batch_data2 = next(self.iter2)
        except StopIteration:
            # Reset iterator if exhausted
            self.iter2 = iter(self.dataloader2)
            batch_data2 = next(self.iter2)

        self.steps += 1
        return batch_data1, batch_data2
