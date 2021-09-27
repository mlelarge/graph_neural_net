import networkx
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from toolbox.utils import check_dir
from matplotlib.colors import ListedColormap

#Creation of colormap
def get_alpha_cmap(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    return ListedColormap(my_cmap)

def show_adjacency(adj,path='pics/',name='graph'):
    check_dir(path)
    n,_ = adj.shape
    g = networkx.empty_graph(n)
    for i in range(n):
        for j in range(n):
            if adj[i,j]==1:
                g.add_edge(i,j)
    plt.figure()
    networkx.draw(g,node_size=50)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
    plt.close()

def show_proba(xs,ys,proba,path='pics/',name='graph',transparency=True,edge_true_range=False):
    n,_ = proba.shape
    assert len(xs)==n and len(ys)==n, f"Length of point coordinates ({len(xs),len(ys)}) not matching size of probability matrix ({proba.shape})"

    g = networkx.empty_graph(n)
    for i in range(0,n-1):
        for j in range(i+1,n):
            g.add_edge(i,j,weight = proba[i,j])

    edges = g.edges()
    colors = [g[u][v]['weight'].item() for u,v in edges]  
    pos = {i:(xs[i],ys[i]) for i in range(n)}
    plt.figure()
    cmap = plt.cm.get_cmap('jet')
    if transparency: cmap = get_alpha_cmap(cmap)
    options = {
        "pos": pos,
        "edge_color": colors,
        "width": 2,
        "edge_cmap": cmap,
        "with_labels": True,
        "node_size": 10,
        'edge_vmin': 0,
        'edge_vmax': 1
    }
    if edge_true_range:
        options['edge_vmin'] = torch.min(proba).item()
        options['edge_vmax'] = torch.max(proba).item()
    networkx.draw(g,**options)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG",dpi=1000)
    plt.close()

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

def show_tour(xs,ys,adj,path='pics/',name='graph', **kwargs):
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
    networkx.draw(g,pos,edge_color = colors,node_size=50,**kwargs)
    plt.tight_layout()
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
    plt.close()

def show_proba_spring(probas,solution=None,path='pics/',name='graph', iterations=1000):
    check_dir(path)

    if solution==None:
        solution = torch.zeros_like(probas)

    n,_ = probas.shape
    #probas = torch.sigmoid(probas)
    g = networkx.empty_graph(n)
    g_sol = networkx.empty_graph(n)
    for i in range(0,n-1):
        for j in range(i+1,n):
            g.add_edge(i,j,weight = torch.exp(-1/probas[i,j]))
            if solution[i,j] or solution[j,i]:
                g_sol.add_edge(i,j)

    edges = g.edges()
    colors = [g[u][v]['weight'].item() for u,v in edges]  

    pos = networkx.spring_layout(g,iterations=iterations)
    plt.figure()
    options = {
        "pos": pos,
        "edge_color": colors,
        "width": 2,
        "edge_cmap": get_alpha_cmap(plt.cm.get_cmap("hot")),
        "with_labels": True,
        "node_size": 10
    }
    options_sol = {
        "pos": pos,
        'width': 1,
        'node_size':0
    }
    networkx.draw(g,**options)
    networkx.draw(g_sol,**options_sol)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG",dpi=1000)
    plt.close()

def show_probas(probs,solution=None,inf_sol_nodes=None,path='pics/',name='graph'):
    check_dir(path)
    if solution is None:
        solution = torch.zeros_like(probs)
        sol_nodes = {}
    elif isinstance(solution,set):
        sol_nodes = solution
    else:
        sol_nodes = torch.where(torch.sum(solution,dim=1))[0] #Sum over rows, then keep the ones that have a degree
        sol_nodes = {elt.item() for elt in sol_nodes}
    if inf_sol_nodes is None:
        inf_sol_nodes = {}

    n,_ = probs.shape
    g = networkx.empty_graph(n)
    for i in range(1,n-1):
        for j in range(i+1,n):
            g.add_edge(i,j,weight = torch.exp(-1/probs[i,j]))

    largest_component = max(networkx.connected_components(g), key=len) #Keep only the largest_component (usually no need for dense graph)
    g2 = g.subgraph(largest_component)
    edges = g.edges()
    colors = [g[u][v]['weight'].item() for u,v in edges]  

    color_code = {
        (True,False): 'red', #Target Solution only
        (False,True): 'blue', #Inferred Solution only
        (True,True): 'green', #Both
        (False,False): 'black' #None
    }
    node_color = [color_code[(elt in sol_nodes,elt in inf_sol_nodes)] for elt in g2.nodes]
    #print([min(1e9,1/probs[u,v].item()) for u,v in edges])
    #pos=networkx.kamada_kawai_layout(g2,[min(1e9,probs[u,v].item()) for u,v in edges])
    pos = networkx.spring_layout(g2)
    plt.figure()
    options = {
        "pos": pos,
        "node_color": node_color,
        "edge_color": colors,
        "width": 2,
        "edge_cmap": get_alpha_cmap(plt.cm.get_cmap("hot")),
        "with_labels": False,
        "node_size": 10
    }
    networkx.draw(g2,**options)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG",dpi=1000)
    plt.close()

def show_probas_sbm(probs,solution=None,inf_sol_nodes=None,path='pics/',name='graph'):
    """
    adj=adjacency matrix, should be of size (n,n) and assumed symmetric, with probabilities on edges
    """
    check_dir(path)
    if solution is None:
        solution = torch.zeros_like(probs)
        sol_nodes = {}
    else:
        sol_nodes = torch.where(torch.sum(solution,dim=1))[0] #Sum over rows, then keep the ones that have a degree
        sol_nodes = {elt.item() for elt in sol_nodes}
    if inf_sol_nodes is None:
        inf_sol_nodes = {}


    n,_ = probs.shape
    g = networkx.empty_graph(n)
    for i in range(1,n-1):
        for j in range(i+1,n):
            g.add_edge(i,j,weight = torch.exp(-1/probs[i,j]))

    largest_component = max(networkx.connected_components(g), key=len) #Keep only the largest_component (usually no need for dense graph)
    g2 = g.subgraph(largest_component)
    edges = g.edges()
    colors = [g[u][v]['weight'].item() for u,v in edges]  

    color_code = {
        (True,False): 'red', #Target Solution only
        (False,True): 'blue', #Inferred Solution only
        (True,True): 'green', #Both
        (False,False): 'black' #None
    }
    node_color = [color_code[(elt in sol_nodes,elt in inf_sol_nodes)] for elt in g2.nodes]
    #print([min(1e9,1/probs[u,v].item()) for u,v in edges])
    #pos=networkx.kamada_kawai_layout(g2,[min(1e9,probs[u,v].item()) for u,v in edges])
    pos = networkx.spring_layout(g2)
    plt.figure()
    options = {
        "pos": pos,
        "node_color": node_color,
        "edge_color": colors,
        "width": 2,
        "edge_cmap": get_alpha_cmap(plt.cm.get_cmap("hot")),
        "with_labels": False,
        "node_size": 10
    }
    networkx.draw(g2,**options)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG",dpi=1000)
    plt.close()

def get_tsne(raw_scores):
    embeddings = F.normalize(raw_scores,dim = -1) #Normalize the embedding vector
    tsne = TSNE()
    v = tsne.fit_transform(embeddings.cpu().detach().squeeze().numpy())
    return v

def show_tsne(raw_scores,target,name='tsne',path='pics/'):
    check_dir(path)
    probas = F.normalize(raw_scores)
    points = get_tsne(probas)
    l_x = points[:,0]
    l_y = points[:,1]
    labels = target[0]
    c = ["blue" if elt==1 else "red" for elt in labels]

    plt.figure()
    plt.scatter(l_x,l_y,c=c)
    plt.savefig(path + name + ".png", format="PNG")
    plt.close()

def show_tsne_bis(raw_scores,target,fname='tsne',dir='pics/'):
    """
    raw_scores and target should be shape (n,n)
    """
    n,_ = raw_scores.shape
    check_dir(dir)
    normalized = F.normalize(raw_scores)
    points = get_tsne(normalized)
    l_x = points[:,0]
    l_y = points[:,1]
    labels = target[0]
    node_color = ["blue" if elt==1 else "red" for elt in labels]

    g = networkx.empty_graph(n)
    for i in range(1,n-1):
        for j in range(i+1,n):
            g.add_edge(i,j,weight = torch.exp(-1/raw_scores[i,j]))
    colors = [g[u][v]['weight'].item() for u,v in g.edges()]
    pos = {i: (l_x[i], l_y[i]) for i in range(n)}
    
    options = {
        "pos": pos,
        "node_color": node_color,
        "edge_color": colors,
        "width": 2,
        "edge_cmap": get_alpha_cmap(plt.cm.get_cmap("hot")),
        "with_labels": False,
        "node_size": 10
    }
    plt.figure()
    networkx.draw(g,**options)
    plt.savefig(dir + fname + ".png", format="PNG")
    plt.close()

def get_pca(raw_scores):
    embeddings = F.normalize(raw_scores,dim = -1) #Normalize the embedding vector
    pca = PCA(n_components=2)
    v = pca.fit_transform(embeddings.cpu().detach().squeeze().numpy())
    return v,pca

def show_pca(raw_scores,target,grad=None,fname='tsne',dir='pics/'):
    check_dir(dir)
    points, pca = get_pca(raw_scores)
    l_x = points[:,0]
    l_y = points[:,1]
    labels = target[0,0]
    c = ["blue" if elt==1 else "red" for elt in labels]
    
    plt.figure()
    if grad is not None:
        grads = pca.transform(grad) 
        grad_x,grad_y = grads[:,0],grads[:,1]
        plt.quiver(l_x,l_y,grad_x,grad_y,alpha = 0.8)
    plt.scatter(l_x,l_y,c=c)
    plt.savefig(dir + fname + ".png", format="PNG")
    plt.close()

def show_adjacency_sbm(adj,target,path='pics/',name='graph'):
    check_dir(path)
    n,_ = adj.shape
    g = networkx.empty_graph(n)
    for i in range(n):
        for j in range(n):
            if adj[i,j]==1:
                g.add_edge(i,j)
    node_color = ["blue" if elt==1 else 'red' for elt in target[0]]
    plt.figure()
    networkx.draw(g,node_size=50,node_color=node_color)
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
    plt.close()
