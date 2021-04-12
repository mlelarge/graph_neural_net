from random import seed
import networkx as nx
import torch
import timeit
import tqdm

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


def N(v, g):
    return [i for i, n_v in enumerate(g[v]) if n_v]

def bronk2(R, P, X, g):
    if not any((P, X)):
        yield R
    for v in P[:]:
        R_v = R + [v]
        P_v = [v1 for v1 in P if v1 in N(v, g)]
        X_v = [v1 for v1 in X if v1 in N(v, g)]
        for r in bronk2(R_v, P_v, X_v, g):
            yield r
        P.remove(v)
        X.append(v)

def find_maxclique(g,ind=None,clique_size=0):
    l_clique = []
    if ind is None:
        ind = range(g.shape[0])
    visited = set()
    l = clique_size
    for x in ind:
        #print(x,visited)
        if not (x in visited):
            for clique in bronk2([x],N(x,g),[],g):
                #print(set(clique))
                if len(clique) > l-1:
                    clique.sort()
                    #print(clique)
                    l = len(clique)
                    visited.update(clique)
                    l_clique.append(clique)
                    #print(visited)
    return l_clique

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
                t_clique = torch.tensor(clique)
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

if __name__ == "__main__":
    g = nx.erdos_renyi_graph(50,0.7)
    adj = torch.tensor(nx.adjacency_matrix(g).todense())
    #l_time = timeit.repeat(lambda : find_maxclique(adj),number=1,repeat=1)

