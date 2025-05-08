import numpy as np
from deap import base,creator,tools,algorithms
import matplotlib.pyplot as plt
import random

#constants
# Define constants
IND_SIZE = 20 # Size of the individual
POP_SIZE = 100  # Population size
CX_PROB = 0.5  # Crossover probability
MUT_PROB = 0.2  # Mutation probability
N_GEN = 10  # Number of generations

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return (sum(individual),)  # Maximizing the number of ones

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB,
                                   ngen=N_GEN, stats=stats, halloffame=hof, verbose=True)
    
    # Find optimal generation
    max_fit = log.select("max")
    optimal_gen = next((i for i, v in enumerate(max_fit) if v == IND_SIZE), N_GEN)
    print(f"Optimal Generation: {optimal_gen}")
    
    # Plot results
    gen = log.select("gen")
    max_fit = log.select("max")
    plt.plot(gen, max_fit, label="Max Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Progress")
    plt.legend()
    plt.show()
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Best Individual:", hof[0])