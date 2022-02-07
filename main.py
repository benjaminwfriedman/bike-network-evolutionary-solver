import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import random
import shapely.speedups
import numpy
import statistics
shapely.speedups.enable()
from multiprocessing import Pool
import concurrent.futures
from evolver import Evolver



if __name__ == "__main__":

    evolver = Evolver()
    evolver.setup_population(3)
    avg_fitnesses = evolver.evolve(3)

    fitness_series = pd.Series(avg_fitnesses)
    print('finished')










