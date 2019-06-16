from NeuralNet import NeuralNet
from Connection import Connection

def neuralNetTest():
    nn = NeuralNet(3, 4)
    connections = nn.getConnections()
    nn.insertNewNode(connections[1], 13, 8)
    nn.show()

def main():
    neuralNetTest()

main()
