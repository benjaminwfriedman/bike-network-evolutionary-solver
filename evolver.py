from solution import Solution, create_a_random_solution, create_new_gen
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


class Evolver:

    def __init__(self, center=(34.105320, -118.283067), dist=1000):

        roads = ox.graph_from_point(center, dist=dist, dist_type='bbox', network_type='all_private', simplify=True,
                                    retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
        buildings = ox.geometries_from_point(center, tags={'building': True}, dist=dist)
        nodes, edges = ox.graph_to_gdfs(roads)
        buildings.reset_index(inplace=True)
        nodes.reset_index(inplace=True)
        edges.reset_index(inplace=True)
        self.buildings = buildings.to_crs(2229)
        self.nodes = nodes.to_crs(2229)
        self.edges = self.clean_roads_data(edges)

        print('Data Pulled')

    def clean_roads_data(self, edges):
        values = []
        for v in edges['highway']:
            if isinstance(v, list):
                value = v[1]
            else:
                value = v
            values.append(value)

        edges['highway'] = values

        values = []
        for v in edges['lanes']:
            if isinstance(v, list):
                value = v[0]
            else:
                value = v
            values.append(value)

        edges['lanes'] = values

        values = []
        for v in edges['maxspeed']:
            if isinstance(v, str):
                split_string = v.split(' ')

                value = split_string[0]
                value = int(value)

            else:

                value = v
            values.append(value)

        edges['maxspeed'] = values

        edges['lanes'] = edges['lanes'].fillna('1').astype('int')
        edges['maxspeed'] = edges['maxspeed'].fillna(20).astype('int')

        return edges

    def setup_population(self, population_size):

        ## randomly generate 10 options\
        POPULATION_SIZE = population_size
        self.population = []
        self.populations_history = []

        print('GENERATING INITIAL POPULATION')
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            population_results = [executor.submit(create_a_random_solution, self.edges, self.buildings) for _ in range(0, POPULATION_SIZE)]
            for s in concurrent.futures.as_completed(population_results):
                self.population.append(s.result())


        print(len(self.population))

    def evolve(self, generations):
        print("STARTING EVOLUTION")
        GENERATIONS = generations
        gen = 1
        avg_fitnesses = []
        for _ in range(0, GENERATIONS):
            df_scores = pd.DataFrame()
            scores = []
            indexes = []
            i = 0
            # populations_history.append(population)
            for s in self.population:
                score = s.fitness_score
                scores.append(score)
                indexes.append(i)
                i = i + 1
            df_scores['solution'] = indexes
            df_scores['score'] = scores
            df_scores.set_index('solution', inplace=True)
            df_scores.sort_values(by='score', ascending=False, inplace=True)
            survivers = df_scores.head(round(df_scores.shape[0] * 0.3)).index

            children = []
            #     new_required = [b for b in range(0, len(population) - len(survivers) + 1)]
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                children_results = [executor.submit(create_new_gen, survivers, self.population) for _ in
                                    range(0, len(self.population) - len(survivers) + 1)]

                for c in concurrent.futures.as_completed(children_results):
                    children.append(c.result())

            # print(children_results)
            #
            # print(children)
            #     print(new_children)
            survivers_obj = []
            for s in survivers:
                obj = self.population[s]
                survivers_obj.append(obj)

            new_population = survivers_obj + children
            self.population = new_population

            fitnesses = [x.fitness_score for x in self.population]
            avg_fitness = statistics.mean(fitnesses)
            avg_fitnesses.append(avg_fitness)

            print(f"EVOLVED GENERATION {gen} OF {GENERATIONS}")
            print(f"AVG Fitness = {avg_fitness}")

            gen = gen + 1

        return avg_fitnesses