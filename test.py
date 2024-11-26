import random

i = 1

# Enumerates every fold number in the current train_dataframe
train_folds = [fold for fold in range(1, 11) if fold != i]

print(train_folds)

# Which allows us to randomly select one of those folds -> validation fold
validation_fold = random.choice(train_folds)

print(validation_fold)