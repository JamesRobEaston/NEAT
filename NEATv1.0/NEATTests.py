from NeuralNet import NeuralNet
from Connection import Connection
from NEAT import NEAT

def neuralNetTest():
    genSize = 10
    neat = NEAT(generationSize = genSize, numInputs = 3, numOutputs = 4, nodeMutationChance = 1.0, existingConnectionMutationChance = 0.0, newConnectionMutationChance = 1.0)
    scores = []
    for j in range(10):
        print(j)
        scores = []
        for i in range(genSize):
            scores.append(i + 1)
        newNets = neat.createNextGenerationFromNetworks(scores)
        neat.setNeuralNets(newNets)
    nets = neat.getNeuralNets()
    for net in nets:
        net.show()

def main():
    neuralNetTest()

main()
