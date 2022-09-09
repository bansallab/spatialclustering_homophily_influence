#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:40:08 2022

@author: luzuzek
"""
###############################################################################
## PACKAGES
###############################################################################
" The Social Influence simulator requires these package to be installed "
import networkx as nx
import numpy as np
import random as rn
import math
import pandas as pd
from scipy import stats
import operator
from networkx.algorithms.community import greedy_modularity_communities

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################################
## PARAMETERS
###############################################################################
def parameters():
    """ Set parameters """
    
    realizations = 100    
    Number_of_nodes = 2048
    mean_degree = 6
    slope = 0.05
    cutoff = 0.05
    jump = 0.01

    
    tmax = 1000
    time = [i for i in range(tmax)]
    
    initial_vulnerables = int(Number_of_nodes * 0.1)
    final_vulnerables = int(Number_of_nodes * 0.25)
    
    p_values = np.arange(0.0,1.01,0.1)   
    
    influence_values = np.arange(0.0,0.84,0.02)
    
    rewiring_values = np.arange(0.0,0.25,0.05)


    return Number_of_nodes, mean_degree, slope, jump, realizations, tmax, time, initial_vulnerables, final_vulnerables, p_values, influence_values, rewiring_values, cutoff

###############################################################################
## HESITANCY LEVELS
###############################################################################
def asign_hesitancy(slope, CI, N, influence_values):
    """ Returns the level of hesitancy for each county"""

    porcentajes = CI / N
    mean = - math.log(porcentajes) / slope

    loc, scale = 0, 1 / mean
    data = stats.expon(loc,scale)  
    refusal_data = data.rvs(N)  
    refusal = [value for value in refusal_data]
    percentile = np.percentile(np.array(refusal),95)
    
    multiplier = percentile
    
    influence_scale = [value * multiplier for value in influence_values]    

    return refusal, influence_scale


def asign_opinion(CI, refusal):        
    """ Set protected and vulnerable counties"""    
    
    for i in nodes: graph_random.nodes[i]['hesitancy'] = refusal[i]       
    count = CI
    
    vulnerable = list(dict(sorted(dict(zip(nodes, refusal)).items(), key=operator.itemgetter(1), reverse=True)[:count]).keys())
    for i in vulnerable: graph_random.nodes[i]['Status'] = 'Vulnerable'  

    protected = [i for i in graph_random.nodes if i not in vulnerable]
    for i in protected: graph_random.nodes[i]['Status'] = 'Protected'
    
    return protected, vulnerable

###############################################################################
## MITIGATION STRATEGIES
###############################################################################
def rewiring_edges_strategy(active_node, rewiring_protected_neighbors, protected):
    """ Rewiring Strategy : Protected counties will rewire edges with vulnerable counties to protected counties. 
    That is to say, non-hesitant counties will remove contact with counties promoting hesitancy, and connect with non-hesitancy counties"""
    
    while True:
        
        if len(rewiring_protected_neighbors) == 0: break
    
        selected_origin = rewiring_protected_neighbors[0]
        neighbors = list(graph_random.neighbors(selected_origin))
        rewire_to = [each_node for each_node in protected if (each_node not in neighbors) and (each_node != selected_origin)]     
    
        selected_destination = rn.choice(rewire_to)  
        
        graph_random.add_edge(selected_origin,selected_destination)        
        graph_random.remove_edge(active_node,selected_origin) 
        rewiring_protected_neighbors.pop(0)
    
    return

###############################################################################
## CLUSTER SIZE DISTRIBUTION
###############################################################################
def community_distribution(CF, protected, regular_graph):
    """ Measure the distribution of vulnerable clusters """
    
    graph_modularity = graph_regular.copy()
    graph_modularity.remove_nodes_from(protected)

    if graph_modularity.number_of_edges() != 0:
        cc = greedy_modularity_communities(graph_modularity)
        sizes = [list(x) for x in cc]
        
        sizes_frecuency = [0] * CF
        for each_size in sizes: sizes_frecuency[len(each_size)] += 1 
        sizes_frecuency.pop(0)
        sizes_values = list(range(1,len(sizes_frecuency)+1))
   
    else:
        sizes_frecuency = [0] * CF
        sizes_values = list(range(1,len(sizes_frecuency)+1))
      
    return sizes_frecuency, sizes_values

###############################################################################
# SPATIAL CLUSTERING
###############################################################################
def measure_spatial_clustering(): 
    """ Spatial clustering : traslate the statues of nodes in the random network to the spatial network.
    Then, we construct two lists with the status of the ending nodes of each edge, and measure the
    Pearson Correlation between the lists """
        
    attrib_list1= [graph_random.nodes[n1]['Opinion'] for n1,n2 in graph_random.edges()]
    attrib_list2= [graph_random.nodes[n2]['Opinion'] for n1,n2 in graph_random.edges()]
    random_clustering = stats.pearsonr(attrib_list1, attrib_list2)[0]
    
    l = 0
    for i in graph_regular.nodes: graph_regular.nodes[i]['Opinion'] = graph_random.nodes[l]['Opinion']; l += 1
    
    attrib_list1= [graph_regular.nodes[n1]['Opinion'] for n1,n2 in graph_regular.edges()]
    attrib_list2= [graph_regular.nodes[n2]['Opinion'] for n1,n2 in graph_regular.edges()]
    spatial_clustering = stats.pearsonr(attrib_list1, attrib_list2)[0]
    
    return random_clustering, spatial_clustering


##############################################################################################################################################################
## PROGRAM
##############################################################################################################################################################
N, k_mean, slope, jump, realizations, tmax, time, CI, CF, p_values, influence_values, rewiring_values, cutoff = parameters()
refusal_original, influence_scaled = asign_hesitancy(slope, CI, N, influence_values)

df_SC = pd.DataFrame(columns=['Realization','Network : Probability of rewiring','Strategy : Probability of rewiring','Social Influence','Spatial Clustering','Clustering'])
df_clusters_sizes = pd.DataFrame(columns=['Realization','Network : Probability of rewiring','Strategy : Probability of rewiring','Social Influence','Size','Frecuency'])

for rea in range(0,realizations,1):
    
    refusal = refusal_original.copy()
    rn.shuffle(refusal)

    for p in p_values:
        
        ###########################################################################
        ## BUILT THE NETWORK
        ##########
        graph_regular = nx.watts_strogatz_graph(N, k_mean, 0)
        graph_random = nx.watts_strogatz_graph(N, k_mean, p)
        
        nodes = [each_node for each_node in graph_random.nodes()]
        edges = [(e1,e2) for (e1,e2) in graph_random.edges()]
        
        labels = 0
        nx.set_edge_attributes(graph_random, labels, "rewired")
        
        ###########################################################################
        ## REWIRING PROBABILITY
        ######################
        for rewiring in rewiring_values:

            ###########################################################################
            ## INFLUENCE 
            ######################     
            for influence in influence_scaled:

                time = 0
                protected, vulnerable = asign_opinion(CI, refusal)
                new_vulnerable = vulnerable.copy()

                ###########################################################################
                ## TIME STEPS
                ######################
                while True:
                    
                    ###########################################################################
                    ## STRATEGY
                    ######################
                    for each_vulnerable in new_vulnerable:

                        protected_neighbors = [each_neighbor for each_neighbor in graph_random.neighbors(each_vulnerable) if graph_random.nodes[each_neighbor]['Status'] == 'Protected']
                        rewiring_neighbors = [each_node for each_node in protected_neighbors if rn.random() < rewiring]
                        rewiring_edges_strategy(each_vulnerable, rewiring_neighbors, protected)
                      
                    new_vulnerable.clear()
                    
                    ###########################################################################
                    ## SOCIAL INFLUENCE PROCESS
                    ######################                   
                    protected = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Protected']
                    vulnerable = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Vulnerable']             
                    
                    fraction = [ [each_node, ((sum([(graph_random.nodes[each_neighbor]['hesitancy']) for each_neighbor in graph_random.neighbors(each_node)]) + graph_random.nodes[each_node]['hesitancy']) / (graph_random.degree(each_node)+1))] for each_node in graph_random.nodes ]
                    result = [ [x,y] for x, y in fraction if (influence < y) ] 
                    result.sort(key = lambda x: x[1], reverse = True)

                    vulnerables_increasing = [each_node for each_node, refusal in result if graph_random.nodes[each_node]['Status'] == 'Vulnerable']
                    for each_node in vulnerables_increasing: graph_random.nodes[each_node]['hesitancy'] = graph_random.nodes[each_node]['hesitancy'] + jump
            
                    protected_increasing = [each_node for each_node, refusal in result if (graph_random.nodes[each_node]['Status'] == 'Protected' and graph_random.nodes[each_node]['hesitancy'] + jump < cutoff)]
                    for each_node in protected_increasing: graph_random.nodes[each_node]['hesitancy'] = graph_random.nodes[each_node]['hesitancy'] + jump
                    protected_changing = [each_node for each_node, refusal in result if (graph_random.nodes[each_node]['Status'] == 'Protected' and graph_random.nodes[each_node]['hesitancy'] + jump >= cutoff)]
                    
                    for each_node in protected_changing:
                        
                        if (len(vulnerable) < CF):
                            graph_random.nodes[each_node]['hesitancy'] = graph_random.nodes[each_node]['hesitancy'] + jump
                            graph_random.nodes[each_node]['Status'] = 'Vulnerable'
                            vulnerable.append(each_node)
                            new_vulnerable.append(each_node)
                            
                    protected  = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Protected']
                    vulnerable = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Vulnerable']      
                    
                    time = time + 1 
                    if time == tmax : break 
                    if len(vulnerable) == CF : break 
                
                    
                ###########################################################################
                ## SAVE RESULTS
                ######################
                if len(vulnerable) >= CF :  
                    
                    sizes_frecuency, sizes_values = community_distribution(CF, protected, graph_regular)

                    for each_step in range(len(sizes_values)):
                        df_clusters_sizes = df_clusters_sizes.append({'Realization' : rea, 'Network : Probability of rewiring' : round(p,2), 'Strategy : Probability of rewiring': round(rewiring,3),'Social Influence' : round(influence_values[influence_scaled.index(influence)],3), 'Size' : sizes_values[each_step], 'Frecuency' : sizes_frecuency[each_step]}, ignore_index=True)
                    
                    for each_node in vulnerable: graph_random.nodes[each_node]['Opinion'] = -1
                    for each_node in protected : graph_random.nodes[each_node]['Opinion'] = 1

                    random_clustering, spatial_clustering = measure_spatial_clustering()
                    if (math.isnan(spatial_clustering) != True):
                        df_SC = df_SC.append({'Realization' : rea, 'Network : Probability of rewiring' : round(p,2), 'Strategy : Probability of rewiring': round(rewiring,3),'Social Influence' : round(influence_values[influence_scaled.index(influence)],3), 'Spatial Clustering' : spatial_clustering, 'Clustering' : random_clustering}, ignore_index=True)

                    ###########################################################################
                    ## EXPORT FILES
                    ######################
                    df_clusters_sizes.to_csv('Social-Influence-p-'+str(p)+'-rewiring-'+str(rewiring)+'-cluster_sizes.csv', index = False)
                    df_SC.to_csv('Social-Influence-SC.csv', index = False)
                    
                    
                    
                    
                    
