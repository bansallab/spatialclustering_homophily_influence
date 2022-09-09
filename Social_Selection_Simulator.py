#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:40:08 2022

@author: luzuzek
"""
###############################################################################
## PACKAGES
###############################################################################
" The Social Selection simulator requires these package to be installed "
import networkx as nx
import numpy as np
import random as rn
import math
import pandas as pd
from scipy import stats
import operator
from scipy.spatial import distance
from networkx.algorithms.community import greedy_modularity_communities

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
    
    initial_vulnerables = int(Number_of_nodes * 0.25)
    ###########################################################################
    ## NETWORK : REWIRING  
    p_values = np.arange(0.0,1.01,0.1)
    
    ###########################################################################
    ## HOMOPHILY : define as 1 - distance(specific case) / distance(random scenario)
    homophily_values = np.arange(0.0,0.51,0.01)   
    homophily_random = 1.7
    
    mahalanobi_distance = [round((1 - each_value) * homophily_random, 2) for each_value in homophily_values]
    
    min_distance = mahalanobi_distance[-1]
    max_distance = homophily_random

    # 0 = No strategy / 1 = Random strategy / 2 = Social Selection strategy / 3 = high hesitancy strategy
    index = 0
                    
    return realizations, Number_of_nodes, mean_degree, slope, cutoff, jump, initial_vulnerables, p_values, homophily_values, mahalanobi_distance, min_distance, max_distance, index


###############################################################################
## HESITANCY LEVELS AND TRAIT CORRELATION
###############################################################################
def asign_hesitancy(slope, CI, N):
    """ Returns the level of hesitancy and traits for each county.
    Creates the desire level of correlation between traits and hesitancy """

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
        
    return x,y,hesitancy

###############################################################################
## INITIAL CONDITIONS
###############################################################################
def initial_conditions(N, CI, trait1, trait2, hesitancy):

    nodes = [each_node for each_node in range(N)]
    
    for each_node in nodes: 
        graph_random.nodes[each_node]['hesitancy'] = hesitancy[each_node]
        graph_random.nodes[each_node]['trait1']    = trait1[each_node]
        graph_random.nodes[each_node]['trait2']    = trait2[each_node]                      
        
    average_before = mahalanobis_average()
    protected, vulnerable = asign_opinion(CI, hesitancy)
    
    return average_before, protected, vulnerable

def asign_opinion(CI, hesitancy):        
    """ Set protected and vulnerable counties"""    
    
    for i in nodes: graph_random.nodes[i]['hesitancy'] = hesitancy[i]       
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

    random_clustering = nx.attribute_assortativity_coefficient(graph_random,'Opinion')
  
    l = 0
    for i in graph_regular.nodes: graph_regular.nodes[i]['Opinion'] = graph_random.nodes[l]['Opinion']; l += 1
    spatial_clustering = nx.attribute_assortativity_coefficient(graph_regular,'Opinion')

    return random_clustering, spatial_clustering

###############################################################################
# MITIGATION STRATEGIES
###############################################################################
def parameters_estrategies(graph_random):
    """ Parameters and measures needed for the implementation of the mitigation strategies"""                  
    target_counties = 0.1   
    
    income = [graph_random.nodes[each_node]['trait1'] for each_node in graph_random.nodes]    
    house = [graph_random.nodes[each_node]['trait2'] for each_node in graph_random.nodes] 
    corr = np.cov(income,house)
    vi = np.linalg.inv(corr) 
    
    degrees = sorted(list(graph_random.degree()),key = lambda x: x[1], reverse = True)

    hesitancy_dict = [[each_node,graph_random.nodes[each_node]['hesitancy']] for each_node in graph_random.nodes]
    hesitancy_sorted = sorted(hesitancy_dict,key = lambda x: x[1], reverse = True)
    
    protected = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Protected']
    vulnerable = [each_node for each_node in graph_random.nodes if graph_random.nodes[each_node]['Status'] == 'Vulnerable']
    
    for each_node in vulnerable: graph_random.nodes[each_node]['Opinion'] = 'Vulnerable'
    for each_node in protected : graph_random.nodes[each_node]['Opinion'] = 'Protected'
    
    return target_counties, income, house, corr, vi, degrees, hesitancy_sorted, protected, vulnerable

###############################################################################
def target_high_hesitancy(percent_target,hesitancys_sorted):

    targets = [hesitancys_sorted[i][0] for i in range(int(percent_target * N))]
    vulnerable_targeted = len([1 for node_targeted in targets if graph_random.nodes[node_targeted]['Opinion'] == 'Vulnerable'])    
    for node_targeted in targets: graph_random.nodes[node_targeted]['Opinion'] = 'Protected'
    
    return vulnerable_targeted, 'Target high hesitancy'


###############################################################################
def target_social_selection(percent_target,vi):
    
    for each_node in graph_random.nodes: graph_random.nodes[each_node]['neighboring comfort'] = 0
    for e1,e2 in graph_random.edges:
        x1 = [graph_random.nodes[e1]['trait1'], graph_random.nodes[e1]['trait2']]
        x2 = [graph_random.nodes[e2]['trait1'], graph_random.nodes[e2]['trait2']]
        each_distance = distance.mahalanobis( x1, x2, vi )
        graph_random.nodes[e1]['neighboring comfort'] = graph_random.nodes[e1]['neighboring comfort'] + each_distance
        graph_random.nodes[e2]['neighboring comfort'] = graph_random.nodes[e2]['neighboring comfort'] + each_distance

    index_numerator = [(graph_random.nodes[each_node]['trait1'] + graph_random.nodes[each_node]['trait2'] ) for each_node in graph_random.nodes]
    index_denominator = [graph_random.nodes[each_node]['neighboring comfort'] for each_node in graph_random.nodes] 
    index = [ [ each_node, math.exp(index_numerator[each_node])/index_denominator[each_node] ] for each_node in graph_random.nodes if index_numerator[each_node] > 0 ]  
    index_sorted = sorted(index,key = lambda x: x[1], reverse = True)
    targets = [index_sorted[i][0] for i in range(int(percent_target * N))]
    vulnerable_targeted = len([1 for node_targeted in targets if graph_random.nodes[node_targeted]['Opinion'] == 'Vulnerable'])
    
    for node_targeted in targets: graph_random.nodes[node_targeted]['Opinion'] = 'Protected'

    return vulnerable_targeted, 'Target social selection'

###############################################################################
def target_randomly(percent_target, vulnerable):
    
    vulnerable_targeted = rn.sample(vulnerable, int(percent_target * N))
    for node_targeted in vulnerable_targeted: graph_random.nodes[node_targeted]['Opinion'] = 'Protected'

    return len(vulnerable_targeted), 'Target randomly'



##############################################################################################################################################################
##############################################################################################################################################################
## PROGRAM
##############################################################################################################################################################
##############################################################################################################################################################
realizations, N, k_mean, slope, cutoff, jump, CI, p_values, homophily_values, mahalanobi_distance, min_distance, max_distance, index = parameters()
name_file = ['No_strategy','Target_high_hesitancy','Target_social_selection','Target_randomly']

df_SC = pd.DataFrame(columns=['Realization','Network : Probability of rewiring','Strategy','Social Selection','Spatial Clustering','Clustering'])
df_clusters_sizes = pd.DataFrame(columns=['Realization','Network : Probability of rewiring','Strategy','Social Selection','Size','Frecuency'])

for rea in range(0,realizations,1):
    
    trait1_original, trait2_original, hesitancy_original = asign_hesitancy(slope, CI, N)

    for p in p_values:
            
        ###########################################################################
        ## BUILT THE NETWORK
        ##########
        graph_regular = nx.watts_strogatz_graph(N, k_mean, 0)
        graph_random = nx.watts_strogatz_graph(N, k_mean, p)
        
        degree = [graph_random.degree(each_node) for each_node in graph_random.nodes]
        nodes = [each_node for each_node in graph_random.nodes()]
        edges = [(e1,e2) for (e1,e2) in graph_random.edges()]
        

        ###########################################################################
        ## HOMOPHILY PROCESS
        ##########

        mahala_average, protected, vulnerable = initial_conditions(N, CI, trait1_original, trait2_original, hesitancy_original)
        count, spatial = 0, 1     
        values_measured = []
        cases = ['Increase','Decrease']
        for each_case in cases:
            
            while True:
                
                if ( each_case == 'Decrease' ) and ( mahala_average < min_distance  ) : break
                if ( each_case == 'Increase' ) and ( mahala_average > max_distance  ) : break
               
                nodo1,nodo2 = rn.sample(nodes,2)
                    
                t1_nodo1,t2_nodo1 = graph_random.nodes[nodo1]['trait1'], graph_random.nodes[nodo1]['trait2']
                t1_nodo2, t2_nodo2 = graph_random.nodes[nodo2]['trait1'], graph_random.nodes[nodo2]['trait2']
                graph_random.nodes[nodo1]['trait1'], graph_random.nodes[nodo1]['trait2'] = t1_nodo2, t2_nodo2
                graph_random.nodes[nodo2]['trait1'], graph_random.nodes[nodo2]['trait2'] = t1_nodo1,t2_nodo1
                mahala_average_updated = mahalanobis_average()      
            
                update = 0
                
                if ( each_case == 'Decrease' ) and ( mahala_average > mahala_average_updated ) : update = 1
                if ( each_case == 'Increase' ) and ( mahala_average < mahala_average_updated ) : update = 1
                
                if update == 1 :
                
                    hesitancy1, hesitancy2 = graph_random.nodes[nodo1]['hesitancy'], graph_random.nodes[nodo2]['hesitancy']
                    graph_random.nodes[nodo1]['hesitancy'], graph_random.nodes[nodo2]['hesitancy'] = hesitancy2, hesitancy1
                    
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
                if (count == 50) : break
                    
                ############################################################### 
                # MEASURE PARAMETERS
                ######################
                distance_value = round(mahala_average_updated, 2)
            
                if distance_value in mahalanobi_distance and distance_value not in values_measured:    

                    homophily = homophily_values[mahalanobi_distance.index(distance_value)]
                    values_measured.append(distance_value)

                    ############################################################### 
                    # DISPLAY THE DIFFERENT STRATEGIES 
                    ######################
                    target_counties, income, house, corr, vi, degrees, hesitancy_sorted, protected, vulnerable = parameters_estrategies(graph_random)
                    random_clustering_no_strategy, spatial_clustering_no_strategy = measure_spatial_clustering()
                    
                    if index == 0 : vulnerable_targeted = len(vulnerable); name = 'No strategy'
                    if index == 1 : vulnerable_targeted, name = target_randomly(target_counties, vulnerable)
                    if index == 2 : vulnerable_targeted, name = target_social_selection(target_counties, vi)
                    if index == 3 : vulnerable_targeted, name = target_high_hesitancy(target_counties, hesitancy_sorted)
    
                    random_clustering, spatial_clustering = measure_spatial_clustering()
                    if (math.isnan(spatial_clustering) != True):
                        df_SC = df_SC.append({'Realization' : rea, 'Network : Probability of rewiring' : round(p,2), 'Strategy': name,'Social Selection' : round(homophily,3), 'Spatial Clustering' : spatial_clustering, 'Clustering' : random_clustering, 'Spatial Clustering without strategy' : spatial_clustering_no_strategy}, ignore_index=True)
    
                    sizes_frecuency, sizes_values = community_distribution(CI, protected, graph_regular)
                    for each_step in range(len(sizes_values)):
                        df_clusters_sizes = df_clusters_sizes.append({'Realization' : rea, 'Network : Probability of rewiring' : round(p,2), 'Strategy' : name,'Social Selection' : round(homophily,3), 'Size' : sizes_values[each_step], 'Frecuency' : sizes_frecuency[each_step]}, ignore_index=True)
                        
                    df_clusters_sizes.to_csv('Social-Selection-p-'+str(p)+'-strategy-'+str(name)+'-cluster_sizes-strategy-'+name+'.csv', index = False)
                    df_SC.to_csv('Social-Selection-SC-strategy-'+name_file[index]+'.csv', index = False)










































































