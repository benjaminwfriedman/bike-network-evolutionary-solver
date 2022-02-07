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

class Solution:
    def __init__(self, df_edges, df_buildings, full_df_edges):
        self.crs = 2229
        self.df_edges = df_edges.to_crs(self.crs)
        self.df_buildings = df_buildings.to_crs(self.crs)
        self.full_df_edges = full_df_edges.to_crs(self.crs)
        #

        if self.df_edges.crs != df_buildings.crs:
            print("WARNING CRSs DONT MATCH")

    def fitness(self):
        self.fitness_score = self.buildings_connected() + self.connectivity() - self.lane_cost() - self.speed_cost() - self.grade_cost() - self.money_cost()
        return self.fitness_score

    def buildings_connected(self):
        self.edges_buffer = self.df_edges.buffer(150)
        self.df_buildings_clipped = gpd.clip(self.df_buildings, self.edges_buffer)
        self.buildings_connected_score = self.df_buildings_clipped.shape[0]
        return self.buildings_connected_score

    def connectivity(self):
        ## TODO Implement this later
        return 0

    def lane_cost(self):
        lane_counts = self.df_edges['lanes'].unique()

        s_lane_counts = pd.DataFrame(data=lane_counts, columns=['counts'])
        s_lane_counts.sort_values(by='counts', ascending=False, inplace=True)
        _max = s_lane_counts['counts'].max()
        s_lane_counts['costs'] = _max + 1 - s_lane_counts['counts']
        s_lane_counts.set_index('counts', inplace=True)
        counts_cost_dict = s_lane_counts.to_dict()

        self.df_edges['lane_cost'] = [counts_cost_dict['costs'][lane] for lane in self.df_edges['lanes']]

        self.lane_cost_score = self.df_edges['lane_cost'].sum()

        return self.lane_cost_score

    def speed_cost(self):
        self.speed_cost_score = self.df_edges['maxspeed'].sum()
        return self.speed_cost_score

    def grade_cost(self):
        return 0

    def money_cost(self):
        self.money_cost_score = self.df_edges['length'].sum() * 10
        return self.money_cost_score

    def mutate(self):
        # 10 percent of the time
        if random.randint(0, 100) < 40:
            # randomly select 95% of the network
            full_length = self.df_edges.shape[0]
            self.df_edges = self.df_edges.sample(frac=0.95)
            lost_roads = full_length - self.df_edges.shape[0]
            # get new roads to replace them
            new_roads = self.full_df_edges.sample(lost_roads)
            self.df_edges.append(new_roads)

            print("mutant")

        return

    def birth(self):
        baby = Solution(self.df_edges, self.df_buildings, self.full_df_edges)
        baby.mutate()
        return baby


def create_new_gen(survivers, population):
    # select a parent
    try:
        print("Creating a New Child")
        parent = numpy.random.choice(survivers)
        child = population[parent].birth()
        child.fitness()
        return child
    except Exception as e:
        raise e

def create_a_random_solution(edges, buildings):
    solution = Solution(df_edges=edges.sample(random.randint(1, 2631)), df_buildings=buildings, full_df_edges=edges)
    solution.fitness()
    return solution


