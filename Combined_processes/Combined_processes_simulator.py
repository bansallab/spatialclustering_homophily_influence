#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:48:10 2022

@author: lucilina
"""

###############################################################################
## PACKAGES
###############################################################################
" The simulator requires these package to be installed "
import networkx as nx
import numpy as np
import random as rn
import math
import pandas as pd
from scipy import stats
import operator
from scipy.spatial import distance
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
    
    initial_vulnerables = int(Number_of_nodes * 0.1)
    final_vulnerables = int(Number_of_nodes * 0.25)
    
    ###########################################################################
    ## NETWORK : REWIRING  
    p_values = [0.0,0.2,0.4,0.6,0.8,1.0]
    
    ###########################################################################
    ## SOCIAL SELECTION : define as 1 - distance(specific case) / distance(random scenario)
    homophily_values = np.arange(0.0,0.51,0.01)   
    homophily_random = 1.7
    
    mahalanobi_distance = [round((1 - each_value) * homophily_random, 2) for each_value in homophily_values]
    
    min_distance = mahalanobi_distance[-1]
    max_distance = homophily_random

    # 0 = No strategy / 1 = Random strategy / 2 = Social Selection strategy / 3 = high Hesitancy strategy
    index = 0
    
    mahalanobi_distance_to_measure = [round((1 - each_value) * homophily_random, 2) for each_value in [0.15,0.25,0.35,0.4,0.45]]

    ###########################################################################
    ## SOCIAL INFLUENCE
    tmax = 1000
    time = [i for i in range(tmax)]

    influence_values = [0.2,0.6]
    
    return realizations, Number_of_nodes, mean_degree, slope, cutoff, jump, initial_vulnerables, final_vulnerables, homophily_values, mahalanobi_distance, \
        min_distance, max_distance, index, tmax, time, influence_values, p_values, mahalanobi_distance_to_measure

###############################################################################
## Hesitancy LEVELS AND TRAIT CORRELATION
###############################################################################
def asign_hesitancy(slope, CI, N):
    """ Returns the level of Hesitancy and traits for each county.
    Creates the desire level of correlation between traits and Hesitancy """

    gamma1 = 3
    gamma2 = 14    
    desire_correlation = 0.0    # NO CORRELATION
    #desire_correlation = 0.7   # MODERATE STRONG
    #desire_correlation = 0.5   # MODERATE WEAK
    #desire_correlation = 0.3   # WEAK

    porcentajes = CI / N
    mean = - math.log(porcentajes) / slope
    data = stats.expon(0, 1 / mean)  
    hesitancy = list(data.rvs(N))

    x = [ hesitancy[each_value] ** (1 / gamma1) for each_value in range(N) ]
    y = [ hesitancy[each_value] ** (1 / gamma2) for each_value in range(N) ]
    
    correlation = np.corrcoef(x, hesitancy)
    nodes = [each_node for each_node in range(N)]
    while True:
        
        if (desire_correlation == 0): break
        if (correlation[0][1] <= desire_correlation): break
        
        node1, node2 = rn.sample(nodes,2)
        x[node1], x[node2] = x[node2], x[node1]
        y[node1], y[node2] = y[node2], y[node1]
        correlation = np.corrcoef(x, hesitancy)

    percentile = np.percentile(np.array(hesitancy),95)
    multiplier = percentile
    influence_scale = [value * multiplier for value in influence_values]    

    return x, y, hesitancy, influence_scale

###############################################################################
## INITIAL CONDITIONS
###############################################################################
def initial_conditions(N, CI, trait1, trait2, hesitancy):

    nodes = [each_node for each_node in range(N)]
    
    for each_node in nodes: 
        graph_random.nodes[each_node]['Hesitancy'] = hesitancy[each_node]
        graph_random.nodes[each_node]['trait1']    = trait1[each_node]
        graph_random.nodes[each_node]['trait2']    = trait2[each_node]                      
        
    average_before = mahalanobis_average()
    protected, vulnerable = asign_opinion(CI, hesitancy)
    
    return average_before, protected, vulnerable

def asign_opinion(CI, hesitancy):        
    """ Set protected and vulnerable counties"""    
    
    for i in nodes: graph_random.nodes[i]['Hesitancy'] = hesitancy[i]       
    count = CI
    
    vulnerable = list(dict(sorted(dict(zip(nodes, hesitancy)).items(), key=operator.itemgetter(1), reverse=True)[:count]).keys())
    for i in vulnerable: graph_random.nodes[i]['Status'] = 'Vulnerable'  

    protected = [i for i in graph_random.nodes if i not in vulnerable]
    for i in protected: graph_random.nodes[i]['Status'] = 'Protected'
    
    return protected, vulnerable

###############################################################################
## HOMOPHILY 
###############################################################################
def mahalanobis_average(): 
    
    xi, xj = [], []
    for i,j in graph_random.edges:
        xi.append([graph_random.nodes[i]['trait1'], graph_random.nodes[i]['trait2']])
        xj.append([graph_random.nodes[j]['trait1'], graph_random.nodes[j]['trait2']])   
    income = [graph_random.nodes[i]['trait1'] for i in graph_random.nodes]    
    house = [graph_random.nodes[i]['trait2'] for i in graph_random.nodes] 
    corr = np.cov(income,house)
    vi = np.linalg.inv(corr)        
    
    distances = [ distance.mahalanobis( xi[i], xj[i], vi ) for i in range(len(xj)) ]
    average_distance = sum(distances)/len(distances)
    
    return average_distance


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

    random_clustering = nx.attribute_assortativity_coefficient(graph_random,'Status')
  
    l = 0
    for i in graph_regular.nodes: graph_regular.nodes[i]['Status'] = graph_random.nodes[l]['Status']; l += 1
    spatial_clustering = nx.attribute_assortativity_coefficient(graph_regular,'Status')

    return random_clustering, spatial_clustering

##############################################################################################################################################################
##############################################################################################################################################################
## PROGRAM
##############################################################################################################################################################
##############################################################################################################################################################
realizations, N, k_mean, slope, cutoff, jump, CI, CF, homophily_values, mahalanobi_distance, min_distance, max_distance, index, tmax, time, influence_values, \
    p_values, mahalanobi_distance_to_measure = parameters()

df_SC = pd.DataFrame(columns=['Realization','Social Selection','Social Influence','Spatial Clustering','Clustering'])


###########################################################################
## NETWORK
########## 
for p in p_values:

    graph_regular = nx.watts_strogatz_graph(N, k_mean, 0)
    graph_random = nx.watts_strogatz_graph(N, k_mean, p)
            
    degree = [graph_random.degree(each_node) for each_node in graph_random.nodes]
    nodes = [each_node for each_node in graph_random.nodes()]
    edges = [(e1,e2) for (e1,e2) in graph_random.edges()]
    
    trait1_original, trait2_original, hesitancy_original, influence_scaled = asign_hesitancy(slope, CI, N)
    
    desire_social_selection = 0.15
    desire_mahalanobi_distance = round((1 - desire_social_selection) * 1.7, 2)
    
    for selection_runs in range(0,realizations,1):

        ###########################################################################
        ## HOMOPHILY PROCESS
        ##########
        mahala_average, protected, vulnerable = initial_conditions(N, CI, trait1_original, trait2_original, hesitancy_original)
        count, spatial = 0, 1     
    
        if mahala_average > max(mahalanobi_distance_to_measure) : case = 'Decrease' 
        if mahala_average < max(mahalanobi_distance_to_measure) : case = 'Increase'
        
        values_measured = []
              
        while True:
            
            print(mahala_average)

            if ( case == 'Decrease' ) and ( mahala_average < min(mahalanobi_distance_to_measure) ) : break
            if ( case == 'Increase' ) and ( mahala_average > max(mahalanobi_distance_to_measure) ) : break      

            nodo1,nodo2 = rn.sample(nodes,2)
                        
            t1_nodo1,t2_nodo1 = graph_random.nodes[nodo1]['trait1'], graph_random.nodes[nodo1]['trait2']
            t1_nodo2, t2_nodo2 = graph_random.nodes[nodo2]['trait1'], graph_random.nodes[nodo2]['trait2']
            graph_random.nodes[nodo1]['trait1'], graph_random.nodes[nodo1]['trait2'] = t1_nodo2, t2_nodo2
            graph_random.nodes[nodo2]['trait1'], graph_random.nodes[nodo2]['trait2'] = t1_nodo1,t2_nodo1
            mahala_average_updated = mahalanobis_average()      
    
            update = 0
            if ( case == 'Decrease' ) and ( mahala_average > mahala_average_updated ) : update = 1
            if ( case == 'Increase' ) and ( mahala_average < mahala_average_updated ) : update = 1
    
            if update == 1 :
            
                hesitancy1, hesitancy2 = graph_random.nodes[nodo1]['Hesitancy'], graph_random.nodes[nodo2]['Hesitancy']
                graph_random.nodes[nodo1]['Hesitancy'], graph_random.nodes[nodo2]['Hesitancy'] = hesitancy2, hesitancy1
                
                mahala_average = mahala_average_updated
                count = 0
                cambio = 1
            else:
                graph_random.nodes[nodo1]['trait1'], graph_random.nodes[nodo1]['trait2'] = t1_nodo1, t2_nodo1
                graph_random.nodes[nodo2]['trait1'], graph_random.nodes[nodo2]['trait2'] = t1_nodo2, t2_nodo2
                cambio = 0 
    
            ############################################################### 
            # IN CASE THE SYSTEM GET STUCK : RESTART
            ######################
            if (round(mahala_average,3) == round(mahala_average_updated,3)) :
                count = count + 1
            else:
                count = 0
            if (count == 100) : break
            
            ############################################################### 
            # REACH DESIRED SOCIAL SELECTION
            ######################
            distance_value = round(mahala_average_updated, 2)   
            if (distance_value in mahalanobi_distance_to_measure) and (distance_value not in values_measured) :             
                
                values_measured.append(distance_value)
                social_selection_value = homophily_values[mahalanobi_distance.index(distance_value)]

                graph_influence = graph_random.copy()
                hesitancy_influence = hesitancy_original.copy()
                
                ############################################################### 
                # 
                ######################        
                for influence in influence_scaled:

                    time = 0
                    protected, vulnerable = asign_opinion(CI, hesitancy_influence)
        
                    ###########################################################################
                    ## TIME STEPS
                    ######################
                    while True:
                        
                        time = time + 1 
                        ###########################################################################
                        ## SOCIAL INFLUENCE PROCESS
                        ######################                   
                        protected = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Protected']
                        vulnerable = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Vulnerable']             
                        
                        fraction = [ [each_node, ((sum([(graph_random.nodes[each_neighbor]['Hesitancy']) for each_neighbor in graph_random.neighbors(each_node)]) + graph_random.nodes[each_node]['Hesitancy']) / (graph_random.degree(each_node)+1))] for each_node in graph_random.nodes ]
                        result = [ [x,y] for x, y in fraction if (influence < y) ] 
                        result.sort(key = lambda x: x[1], reverse = True)
                
                        vulnerables_increasing = [each_node for each_node, refusal in result if graph_random.nodes[each_node]['Status'] == 'Vulnerable']
                        for each_node in vulnerables_increasing: graph_random.nodes[each_node]['Hesitancy'] = graph_random.nodes[each_node]['Hesitancy'] + jump
                
                        protected_increasing = [each_node for each_node, refusal in result if (graph_random.nodes[each_node]['Status'] == 'Protected' and graph_random.nodes[each_node]['Hesitancy'] + jump < cutoff)]
                        for each_node in protected_increasing: graph_random.nodes[each_node]['Hesitancy'] = graph_random.nodes[each_node]['Hesitancy'] + jump
                        protected_changing = [each_node for each_node, refusal in result if (graph_random.nodes[each_node]['Status'] == 'Protected' and graph_random.nodes[each_node]['Hesitancy'] + jump >= cutoff)]
                
                        for each_node in protected_changing:
                            
                            if (len(vulnerable) < CF):
                                graph_random.nodes[each_node]['Hesitancy'] = graph_random.nodes[each_node]['Hesitancy'] + jump
                                graph_random.nodes[each_node]['Status'] = 'Vulnerable'
                                vulnerable.append(each_node)
                                
                        protected  = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Protected']
                        vulnerable = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Vulnerable']      
                        
                        if time == tmax : break 
                        if len(vulnerable) == CF : break 
                

                    ###########################################################################
                    ## SAVE RESULTS
                    ######################
                    if len(vulnerable) >= CF :  

                        random_clustering, spatial_clustering = measure_spatial_clustering()
                        if (math.isnan(spatial_clustering) != True):
                            df_SC = df_SC.append({'Network : rewiring probability' : p, 'Realization' : selection_runs, 'Social Selection' : social_selection_value, 'Social Influence' : round(influence_values[influence_scaled.index(influence)],3), 'Spatial Clustering' : spatial_clustering, 'Clustering' : random_clustering}, ignore_index = True)
    
                        ###########################################################################
                        ## EXPORT FILE
                        ######################
                        df_SC.to_csv('Combined-Processes-SC.csv')
    
