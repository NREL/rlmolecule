import pickle
import numpy as np
import random

if __name__ == "__main__":

    sample = random.sample(range(10),2)
    
    for i in sample:
        with open('game_{}.pickle'.format(i), 'rb') as f:
            new_data = pickle.load(f)
            print(new_data)