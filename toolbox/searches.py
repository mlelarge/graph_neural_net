from random import seed
import networkx as nx
from networkx.readwrite.json_graph import adjacency
import torch
import numpy as np
import toolbox.utils as utils
from toolbox.mcp_solver import MCP_Solver
import tqdm
from scipy.spatial.distance import pdist, squareform
import random
import time
import string
import os



def insert(container, new_item, key=len):
    """
    Just a dichotomy to place an item into a tensor depending on a key, supposing the list is ordered in a decresc manner
    """
    if len(container)==0:
        return [new_item]

    l,r = 0, len(container)
    item_value = key(new_item)
    while l!=r:
        mid = (l+r)//2
        if key(container[mid])>=item_value:
            l=mid+1
        else:
            r = mid
    return container[:l] + [new_item] + container[l:]

#MCP

def neighs(v,adj):
    return {i for i in range(adj.shape[0]) if adj[v,i]}


def _bronk2(R,P,X,adj):
    if len(P)==0 and len(X)==0:
        yield R
    else:
        u = random.sample(P.union(X),1)
        N_u = neighs(u,adj)
        for v in P-N_u:
            N_v = neighs(v,adj)
            for clique in _bronk2(R.union({v}),P.intersection(N_v),X.intersection(N_v),adj):
                yield clique
            P = P - {v}
            X = X.union({v})

def mc_bronk2(adj):
    assert (adj==(adj.T+adj)/2).all(), "Matrix is not symmetric"
    n,_ = adj.shape
    adj = adj * (1 - torch.diag_embed(torch.ones(n)))
    base_set = {i for i in range(n)}
    max_cliques = []
    max_length=0
    for c in _bronk2(set(),base_set,set(),adj):
        cur_l = len(c)
        if cur_l==max_length:
            max_cliques.append(c)
        elif cur_l>max_length:
            max_cliques = [c]
            max_length=cur_l
    return max_cliques

def write_adj(fname,adj):
    with open(fname,'w') as f:
        for row in adj:
            line = ""
            for value in row:
                line+=f"{int(value)} "
            line = line[:-1] + "\n"
            f.write(line)

def read_adj(fname):
    with open(fname,'r') as f:
        data = f.readlines()
    cliques = []
    for i,line in enumerate(data):
        cur_data = {int(elt) for elt in line.split(' ')}
        cliques.append(cur_data)
    return cliques

def mc_bronk2_cpp(adjs,max_threads=4):
    """
    adj should be of shape (bs,n,n) or (n,n)
    """
    path = "tmp_mcp/"
    utils.check_dir(path)
    solver = MCP_Solver(adjs,max_threads=max_threads)
    solver.solve()
    clique_sols = solver.solutions
    return clique_sols


def mcp_proba_cheat(data,raw_scores, solutions, overestimate=10):
    """
    data should be (bs,in_features,n,n) with data[:,:,:,1] the adjacency matrices
    raw_scores and solutions should be (bs,n,n)
    Searches the max clique among the ' n = clique size + overestimate ' best nodes, then completes with the rest
    """
    adjs = data[:,:,:,1]
    clique_sizes,_ = torch.max(solutions.sum(dim=-1),dim=-1) 
    clique_sizes += 1 #The '+1' is because the diagonal of the solutions is 0
    bs,n,_ = raw_scores.shape

    probas = torch.sigmoid(raw_scores)

    sol_onehot = torch.sum(solutions,dim=-1)#Gets the onehot encoding of the solution clique
    degrees = torch.sum(probas, dim=-1)
    inds = [ (torch.topk(degrees[k],int(clique_sizes[k].item() + overestimate),dim=-1))[1] for k in range(bs)]

    l_clique_sol = []
    l_clique_inf = []
    
    for i in tqdm.tqdm(range(len(degrees)),desc='Max Clique Search counter'):
        search_inds = [elt.item() for elt in inds[i]]
        l_cliques = find_maxclique(adjs[i],ind=search_inds) #Could be quite lengthy
        inf_clique_size = max([len(clique) for clique in l_cliques]) #Save max clique_size
        best_sets = [set(clique) for clique in l_cliques if len(clique) == inf_clique_size] #Transform the best cliques in sets
        
        cur_sol_nodes = torch.where(sol_onehot[i])[0] #Converts the onehot encoding to a list of the nodes' numbers
        sol_set = {elt.item() for elt in cur_sol_nodes} #Converts to a set

        best_set = max(best_sets, key= lambda set: len(set.intersection(sol_set)) ) #Gets the best set by intersection with the solution
        l_clique_sol.append(sol_set)
        l_clique_inf.append(best_set)
    return l_clique_inf,l_clique_sol

def mcp_beam_method(data, raw_scores, seeds=None, add_singles=True, beam_size=1280):
    """
    The idea of this method is to establish a growing clique, keeping only the biggest cliques starting from the most probable nodes
    seeds should be a list of sets
    """
    seeding = (seeds is not None)

    solo=False
    if len(raw_scores.shape)==2:
        solo=True
        raw_scores = raw_scores.unsqueeze(0)
        data = data.unsqueeze(0)
        if seeding: seeds = [seeds] #In that case we'd only have a set
    
    if len(data.shape)==3:
        data = data.unsqueeze(0)
    

    bs,n,_ = raw_scores.shape

    adjs = data[:,:,:,1]
    probas = torch.sigmoid(raw_scores)

    degrees = torch.sum(probas, dim=-1)
    inds_order = torch.argsort(degrees,dim=-1,descending=True) #Sort them in ascending order
    
    l_clique_inf = []
    for k in range(bs): #For the different data in the batch
        cliques = [] #Will contain 1D Tensors
        cur_adj = adjs[k]
        node_order = torch.arange(n)[inds_order[k]] #Creates the node order
        if seeding:
            seed = seeds[k]
            node_order = [elt.item() for elt in node_order if not elt.item() in seed] #Remove the elements of the seed
            cliques.append(torch.tensor([elt for elt in seed]))
        for cur_step in range(len(node_order)):
            cur_node = node_order[cur_step]
            for clique in cliques: #Iterate over the currently saved cliques to make them grow
                t_clique = clique.clone().detach()
                neighs = cur_adj[cur_node][t_clique]
                if torch.all(neighs==1): #If all clique nodes are adjacent to cur_node
                    new_clique = torch.cat((clique,torch.tensor([cur_node],dtype=torch.long)))
                    cliques = insert(cliques,new_clique)
            if add_singles: cliques = insert(cliques,torch.tensor([cur_node])) #Add the clique with just the node
            cliques = cliques[:beam_size] # Keep a good size
        #Now choose one of the best, knowing cliques is ordered descendingly
        #I just choose the first one, but we can choose the most similar to solution ?
        best_set = set([elt.item() for elt in cliques[0]])
        l_clique_inf.append(best_set)
    if solo:
        l_clique_inf = l_clique_inf[0]
    return l_clique_inf

#TSP

def tsp_greedy_decoding(G):
    '''
    Starts from the first node. At every steps, it looks for the most probable neighbors
    which hasn't been visited yet, which yields a tour at the end
    '''
    batch_size,n,_ = G.size()
    output = torch.zeros(batch_size,n,n)
    for k in range(batch_size):
        curr_output = torch.zeros(n)
        current = torch.randint(n,(1,1)).item()
        not_seen = torch.ones(n, dtype=torch.bool)
        not_seen[current] = False
        curr_output[0] = current
        counter = 1
        while counter < n:
            nxt = torch.argmax(G[k][current]*not_seen)
            not_seen[nxt] = False
            curr_output[counter] = nxt
            current = nxt
            counter+=1
            output[k] = utils.tour_to_adj(n,curr_output)
    return output

def get_confused(n,G):
    """
    Gives the 'most-confused' node : the node that has the biggest std of probabilities
    Needs G.shape = n,n
    """
    maxi_std = -1
    node_idx = -1
    for node in range(n):
        cur_node = G[node,:]
        cur_std = cur_node.std()
        if cur_std>maxi_std:
            maxi_std = cur_std
            node_idx = node
    assert node_idx!=-1, "Should not be possible to have std always smaller than -1"
    return node_idx

def get_surest(n,G):
    """
    Gives the 'surest node : the node that has the biggest edge proba
    Needs G.shape = n,n
    """
    node_idx = torch.argmax(G.flatten())//n
    return node_idx

def tsp_beam_decode(raw_scores,l_xs=[],l_ys=[],W_dists=None,b=1280,start_mode="r",chosen=0,keep_beams=0):

    start_mode = start_mode.lower()
    if start_mode=='r':
        start_fn = lambda n, G : torch.randint(n,(1,1)).item()
    elif start_mode=='c':
        start_fn = lambda n, G : chosen
    elif start_mode=='conf': #Confusion
        start_fn = get_confused
    elif start_mode=='sure': #Start from the surest edge
        start_fn = get_surest
    else:
        raise KeyError("Start function {} not implemented.".format(start_mode))

    l_beam = []


    with torch.no_grad(): #Make sure no gradient is computed
        G = torch.sigmoid(raw_scores)

        bs,n,_ = G.shape
        
        output = torch.zeros(bs,n,n)

        diag_mask = torch.diag_embed(torch.ones(bs,n,dtype=torch.bool))
        G[diag_mask] = 0 #Make sure the probability of staying on a node is 0

        for k in range(bs):
            beams = torch.zeros(b,n, dtype=torch.int64)
            beams_score = torch.zeros((b,1))
            cur_g = G[k]
            start_node = start_fn(n,cur_g)
            cur_b = 1
            beams[:1,0] = start_node
            beams_score[:1] = 1
            for beam_time in range(1,n):
                not_seen = torch.ones((cur_b,n), dtype=torch.bool)
                not_seen.scatter_(1,beams[:cur_b,:beam_time],0) # Places False where a beam has already passed
                cur_neigh = cur_g[beams[:cur_b,beam_time-1]] #Love this syntax, just takes the neighbour values for each beam : cur_neigh.shape = (cur_b,n)
                nxt_values, nxt_indices = torch.topk(not_seen*cur_neigh,n,-1)
                nxt_values = nxt_values * beams_score[:cur_b]
                cur_b = min(b,cur_b*n)
                _, best_indices = torch.topk(nxt_values.flatten(), cur_b)
                best = torch.tensor(np.array(np.unravel_index(best_indices.numpy(), nxt_values.shape)).T)
                new_beams = torch.zeros(cur_b,n, dtype=torch.int64)
                for j in range(len(best)):
                    x,y = best[j]
                    new_beams[j,beam_time] = nxt_indices[x,y]
                    new_beams[j,:beam_time] = beams[x,:beam_time]
                    beams_score[j] = nxt_values[x,y]
                beams = new_beams
                assert beams_score[0].item()>0, "Zero in the most probable beam. That's not good."
                beams_score /= beams_score[0].item() #This prevents probabilities from going all the way to 0 by renormalizing the first score to 1
            #Now add last edge to the score
            beams_score = beams_score * torch.unsqueeze(cur_g[beams[:,-1],start_node],-1)
            
            if keep_beams!=0:
                l_beam.append(beams[:keep_beams])

            if W_dists is None:
                xs,ys = l_xs[k],l_ys[k]
                nodes_coord = [ (xs[i],ys[i]) for i in range(len(xs))]
                W_dist = torch.tensor(squareform(pdist(nodes_coord, metric='euclidean')))
            else:
                W_dist = W_dists[k]
            mini = torch.sum(W_dist)
            best_beam_idx = -1
            for beam_num in range(beams.shape[0]):
                beam = beams[beam_num]
                path_length = 0
                for node in range(n):
                    path_length += W_dist[beam[node],beam[(node+1)%n]]
                if path_length<=mini:
                    mini=path_length
                    best_beam_idx = beam_num


            best_beam = beams[best_beam_idx]
            output[k] = utils.tour_to_adj(n,best_beam)

            assert utils.is_permutation_matrix(utils.tour_to_perm(n,best_beam)), "Result of beam_fs is not a permutation !"
    
    if keep_beams!=0:
        return output, l_beam
    return output


if __name__ == "__main__":
    #g = nx.erdos_renyi_graph(50,0.7)
    #adj = torch.tensor(nx.adjacency_matrix(g).todense())
    #l_time = timeit.repeat(lambda : find_maxclique(adj),number=1,repeat=1)
    n=100
    #def time_bronk(n):
    #    g = torch.empty((n,n)).uniform_()
    #    g = (g.T+g)/2
    #    #print(g)
    #    g = (g<(0.9)).to(int)
    #    #print(g)
    #    print(mc_bronk2(g))
    #print(timeit.timeit(lambda : time_bronk(n),number=5))
    
    def test_mcp_solver(bs,n):
        adjs = torch.empty((bs,n,n)).uniform_()
        adjs = (adjs.transpose(-1,-2)+adjs)/2
        adjs = (adjs<(0.5)).to(int)
        clique_sols = mc_bronk2_cpp(adjs)
        return clique_sols
    clique_sols = test_mcp_solver(10,n)
