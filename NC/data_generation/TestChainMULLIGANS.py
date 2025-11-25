#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 7 12:17:40 2021

@author: macbarnes
"""

from gerrychain import Graph, Partition
from gerrychain.updaters import cut_edges, county_splits, Tally
from gerrychain.constraints import contiguity, UpperBound, LowerBound, within_percent_of_ideal_population
from gerrychain.tree import recursive_tree_part
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
import pandas
import json
from enum import Enum
import gerrymetrics as gm
import numpy as np

# COMPUTING COUNTY SPLITS
class CountySplit(Enum):
    NOT_SPLIT = 0
    NEW_SPLIT = 1
    OLD_SPLIT = 2

def calc_splits(dictionary):
    count_splits = 0
    for name, data in dictionary.items():
        if data[0].value != 0:
            count_splits = count_splits + 1
    return count_splits

election_names = [
    "USSEN16",
    "PRES16",
    "GOV16"
]
election_columns = [
    ["G16USSDROS", "G16USSRBUR"],
    ["G16PREDCLI", "G16PRERTRU"],
    ["G16GOVDCOO", "G16GOVRMCC"]
]

num_dist = 13

# IMPORT GRAPH DATA
nc_graph = Graph.from_file(r"/Users/macbarnes/Desktop/Data/NC_precs/NC_precs_all_data.shp")

#UPDATERS
#County_ID is the name of the county column in the shapefile

updaters = { "county_splits": county_splits("county_splits", "COUNTY_ID"),
             "population": Tally("tot", alias="population")
             }

num_elections = len(election_names)
elections = [
    Election(
        election_names[i],
        {"Dem": election_columns[i][0], "Rep": election_columns[i][1]},
    )
    for i in range(num_elections)
]
election_updaters = {election.name: election for election in elections}
updaters.update(election_updaters)


# INITIAL PARTITION
initial_partition = GeographicPartition(nc_graph, assignment="Cong19Dist", updaters=updaters)
# using GeographicPartition lets us compute things like PolsbyPropper and other compactness scores



# Recom Proposal needs to know the ideal population for the districts so that
# we can bail on early unbalanced partitions to increase speed

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)



# PROPOSAL

# we use functools.partial to bind the extra paramenters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal
proposal = partial(recom,
                   pop_col="tot",
                   pop_target=ideal_population,
                   epsilon=0.02, node_repeats=2
                   )


# CONSTRAINTS
compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]),
                                           len(initial_partition["cut_edges"]))

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

county_splits_bound = constraints.UpperBound(lambda p: calc_splits(p["county_splits"]),
                                             calc_splits(initial_partition["county_splits"]))


# CONFIGURING THE MARKOV CHAIN
chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound,
        county_splits_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000
)

# FILING THE METRICS
total_nodes = len(nc_graph.nodes)
list_of_nodes = list(nc_graph.nodes)
cut_edge_array = []
county_splits_array = []
array_of_precinct_assignments = []
PB_array = []

counter = 0


# RUNNING THE CHAIN
for part in chain:
    print(counter)
    if int(counter / 10) == counter / 10:
        cut_edge_array.append(len(part["cut_edges"]))
        county_splits_array.append(calc_splits(part["county_splits"]))
        precinct_identifier = [nc_graph.nodes[y]["PREC_ID"] for y in range(total_nodes)]
        precinct_assignment = [part.assignment[x] for x in range(total_nodes)]
        array_of_precinct_assignments.append(precinct_assignment)
        accumulating_election = np.zeros((1, num_dist))
        for i in election_names:
            vars()["{}part".format(i)] = np.array(part[i].percents("Dem"))
            accumulating_election = accumulating_election + vars()["{}part".format(i)]
        avg = accumulating_election / num_elections
        avg_elect = avg[0]
        partisan_bias_val = gm.partisan_bias(avg_elect)
        PB_array.append(partisan_bias_val)
    counter = counter + 1


# EXTRACTING THE PLANS & METRICS AS .JSON
json_with_data = {
    "EquivalencyFiles": array_of_precinct_assignments,
    "CompactnessScores": cut_edge_array,
    "CountySplitsScores": county_splits_array,
    "PartisanBiasArray": PB_array
}

with open("100testrun1.json", "w") as data_file:
    json.dump(json_with_data, data_file, indent=4)





