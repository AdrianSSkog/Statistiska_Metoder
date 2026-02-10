import pygame as pyg
import numpy as np
import matplotlib.pyplot as plt

def get_graph_data():
   
    rng = np.random.default_rng()

    X1 = rng.integers(100, size=100)
    X2 = rng.uniform(size=100)
    X3 = rng.normal(size=100)
    X4 = rng.binomial(n=10,p=.3,size=100)
    X5 = rng.negative_binomial(n=10,p=.3,size=100)
    X6 = rng.gamma(shape=1.0, scale=0.9, size=100)
    X7 = rng.geometric(p=0.31, size=100)

    XList = [X1, X2, X3, X4, X5, X6, X7] 
    XTypes = ["Integers", "Uniform", "Normal", "Binomial", "Negative binomial", "Gamma", "Geometric"]
    
    idx = rng.integers(len(XList))
    graph_data = XList[idx]
    correct_answer = XTypes[idx]

    plt.figure(figsize=(8,6))
    plt.hist(graph_data, bins=15)
    plt.tight_layout
    plt.savefig("hist.png")
    plt.close()
    
    
    return correct_answer

def new_question(Width):
    correct = get_graph_data()
    image = pyg.image.load("hist.png")
    rect = image.get_rect(center=(Width // 2, 250))
    return correct, image, rect

