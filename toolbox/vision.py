import networkx
import matplotlib.pyplot as plt
import os
import torch

def compare(xs,ys,adj_out,adj_target,save_out=True,path='pics/',name='graph'):
    N = len(xs)
    pos = {i: (xs[i], ys[i]) for i in range(N)}
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g_out = networkx.random_geometric_graph(N,0,pos=pos)
    for i in range(N):
        for j in range(N):
            if 1==adj_out[i,j]==adj_target[i,j]:
                g.add_edge(i,j,color="black")
                g_out.add_edge(i,j,color="black")
            elif adj_out[i,j]:
                g.add_edge(i,j,color="blue")
                g_out.add_edge(i,j,color="blue")
            elif adj_target[i,j]:
                g.add_edge(i,j,color="red")
    
    edges = g.edges()
    colors = [g[u][v]['color'] for u,v in edges]
    plt.figure()
    networkx.draw(g,pos,edge_color = colors,node_size=100)
    plt.tight_layout()
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
    plt.close()
    
    if save_out:
        plt.figure()
        edges = g_out.edges()
        colors = [g_out[u][v]['color'] for u,v in edges]
        networkx.draw(g_out,pos,edge_color = colors,node_size=100)
        plt.tight_layout()
        plt.show()
        fname = os.path.join(path,name)
        plt.savefig(fname+"_out.png", format="PNG")
        plt.close()

def show_tour(xs,ys,adj,path='pics/',name='graph'):
    N = len(xs)
    pos = {i: (xs[i], ys[i]) for i in range(N)}
    g = networkx.random_geometric_graph(N,0,pos=pos)
    for i in range(N):
        for j in range(N):
            if adj[i,j]==1:
                g.add_edge(i,j,color="black")
    edges = g.edges()
    colors = [g[u][v]['color'] for u,v in edges]
    plt.figure()
    networkx.draw(g,pos,edge_color = colors,node_size=50)
    plt.tight_layout()
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
    plt.close()

def show_probas(probs,solution=None,path='pics/',name='graph'):
    """
    adj=adjacency matrix, should be of size (n,n) and assumed symmetric, with probabilities on edges
    """
    if solution is None:
        solution = torch.zeros_like(probs)
        sol_nodes = []
    else:
        sol_nodes = torch.where(torch.sum(solution,dim=1))[0] #Sum over rows, then keep the ones that have a degree
    n,_ = probs.shape
    g = networkx.empty_graph(n)
    black = torch.ones((3))
    maxi = torch.max(probs)
    probs = probs/maxi
    for i in range(1,n):
        for j in range(i,n):
            if solution[i,j]:
                g.add_edge(i,j,color = torch.tensor([1,0,0]), weight = torch.exp(-1/probs[i,j]))
            elif probs[i,j]:
                g.add_edge(i,j,color = (1-probs[i,j])*black,weight = torch.exp(-1/probs[i,j]))
    largest_component = max(networkx.connected_components(g), key=len)
    g2 = g.subgraph(largest_component)
    edges = g.edges()
    colors = [[g[u][v]['color'][i].item() for i in range(3)] for u,v in edges]
    node_color = ['red' if i in sol_nodes else 'blue' for i in g2.nodes]
    #print([min(1e9,1/probs[u,v].item()) for u,v in edges])
    pos = networkx.spring_layout(g2)#networkx.kamada_kawai_layout(g2,[min(1e9,probs[u,v].item()) for u,v in edges])
    #alpha = [probs[u,v].item() for u,v in edges]
    plt.figure()
    networkx.draw(g2,pos=pos,node_color=node_color,edge_color=colors,node_size=5)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG",dpi=1000)
    plt.close()