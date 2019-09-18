from NeuralNet import NeuralNet
from Connection import Connection
from NEAT import NEAT

def testNeuralNetConstructor():
    #Set up the test
    neuralNet = NeuralNet(numInputs = 10, numOutputs = 11)
    inputs = neuralNet.getInputs()
    outputs = neuralNet.getOutputs()
    nodes = neuralNet.getNodes()

    #Ensure the correct number of nodes were created
    success = len(inputs) == 10
    success = success and len(outputs) == 11
    success = success and len(nodes) == 21

    #Ensure the inputs and outputs are connected approriately
    for inputNode in inputs:
        nodeOutputs = inputNode.getOutNodes()
        for i in range(len(outputs)):
            success = success and outputs[i] == nodeOutputs[i]
    for outputNode in outputs:
        nodeInputs = outputNode.getInNodes()
        for i in range(len(inputs)):
            success = success and inputs[i] == nodeInputs[i]
    
    #Ensure the connections have seemingly random values by ensuring they are inconsistent
    neuralNet2 = NeuralNet(numInputs = 10, numOutputs = 11)
    nodes2 = neuralNet2.getNodes()
    for connectionIndex in range(len(inputs)):
        success = success and nodes[connectionIndex].getOutConnections()[0].getWeight() != nodes2[connectionIndex].getOutConnections()[0].getWeight()

    return success

def testCreateNewNode():
    #Set up the test
    neuralNet = NeuralNet(numInputs = 1, numOutputs = 1)
    newNode = neuralNet.createNewNode([], 2, 3)
    nodes = neuralNet.getNodes()

    #Ensure the node exists
    success = newNode != None

    #Ensure the node is in the correct location
    success = success and nodes[1] == newNode

    #Ensure the node has the correct connections
    success = success and newNode.getNumInConnections() == 0
    success = success and newNode.getNumOutConnections() == 0

    #Ensure multiple nodes can be added
    newNode2 = neuralNet.createNewNode([],3,4)
    success = success and newNode2 != None
    success = success and nodes[1] == newNode2

    #Ensure the nodes are not the same
    success = success and newNode != newNode2

    return success

def testCreateNewInput():
    #Set up the test
    neuralNet = NeuralNet(numInputs = 1, numOutputs = 1)
    newNode = neuralNet.createNewInput(3)
    nodes = neuralNet.getNodes()
    inputs = neuralNet.getInputs()

    #Ensure the node exists
    success = newNode != None

    #Ensure the node is in the correct location
    success = success and nodes[0] == newNode
    success = success and inputs[1] == newNode

    #Ensure a new node was added
    success = success and len(nodes) == 3
    success = success and len(inputs) == 2

    #Ensure the node has the correct connections
    success = success and newNode.getNumInConnections() == 0
    success = success and newNode.getNumOutConnections() == 0

    #Ensure multiple nodes can be added
    newNode2 = neuralNet.createNewInput(4)
    success = success and newNode2 != None
    success = success and nodes[0] == newNode2
    success = success and len(inputs) == 3

    return success

def testCreateNewOutput():
    #Set up the test
    neuralNet = NeuralNet(numInputs = 1, numOutputs = 1)
    newNode = neuralNet.createNewOutput(3)
    nodes = neuralNet.getNodes()
    outputs = neuralNet.getOutputs()

    #Ensure the node exists
    success = newNode != None

    #Ensure the node is in the correct location
    success = success and nodes[2] == newNode
    success = success and outputs[1] == newNode

    #Ensure a new node was added
    success = success and len(nodes) == 3
    success = success and len(outputs) == 2

    #Ensure the node has the correct connections
    success = success and newNode.getNumInConnections() == 0
    success = success and newNode.getNumOutConnections() == 0

    #Ensure multiple nodes can be added
    newNode2 = neuralNet.createNewOutput(4)
    success = success and newNode2 != None
    success = success and nodes[3] == newNode2
    success = success and len(outputs) == 3

    #Ensure adding another node affects the position of outputs appropriately
    neuralNet.createNewNode([], 3, 5)
    success = success and nodes[3] == newNode
    success = success and nodes[4] == newNode2    

    return success

def testInsertNewNode():
    #Set up the test
    neuralNet = NeuralNet(numInputs = 1, numOutputs = 1)
    connection = neuralNet.getInputs()[0].getOutConnections()[0]
    newNode = neuralNet.insertNewNode(connection, 2, 3)
    nodes = neuralNet.getNodes()

    #Ensure the node exists
    success = newNode != None

    #Ensure the old connection is disabled
    success = success and not connection.isEnabled()

    #Ensure the node is in the correct location
    success = success and nodes[1] == newNode

    #Ensure a new node was added
    success = success and len(nodes) == 3

    #Ensure the node has the correct connections
    success = success and newNode.getNumInConnections() == 1
    success = success and newNode.getNumOutConnections() == 1

    #Ensure the node is connected to the correct nodes
    success = success and newNode.getInNodes()[0] == neuralNet.getInputs()[0]
    success = success and newNode.getOutNodes()[0] == neuralNet.getOutputs()[0]

    #Ensure multiple nodes can be added
    connection = newNode.getInConnections()[0]
    newNode2 = neuralNet.insertNewNode(connection, 3, 4)
    success = success and newNode2 != None
    success = success and nodes[1] == newNode2

    #Ensure the node is connected to the correct nodes
    success = success and newNode2.getInNodes()[0] == neuralNet.getInputs()[0]
    success = success and newNode2.getOutNodes()[0] == nodes[2]

    #Ensure that a node can be connected between previously inserted nodes
    connection = newNode2.getOutConnections()[0]
    newNode3 = neuralNet.insertNewNode(connection,3,5)
    success = success and newNode3 != None
    success = success and nodes[2] == newNode3

    #Ensure the node is connected to the correct nodes
    success = success and newNode3.getInNodes()[0] == nodes[1]
    success = success and newNode3.getOutNodes()[0] == nodes[3]
    
    return success

def testGetNonexistantConnections():
    #Set up the test
    neuralNet = NeuralNet(numInputs = 5, numOutputs = 5)

    #Ensure there are no nonexistant connections
    success = len(neuralNet.getNonexistantConnections(0)) == 0

    #Insert a node and ensure that the reported nonexistant connections is correct
    neuralNet.insertNewNode(neuralNet.getInputs()[0].getOutConnections()[0], 2, 11)
    nonexistantConnections = neuralNet.getNonexistantConnections(0)
    success = success and len(nonexistantConnections) == 9
    nonexistantPairs = [[2,11],[3,11],[4,11],[5,11],[11,7],[11,8],[11,9],[11,10],[1,6]]

    for i in range(len(nonexistantConnections)):
        connection = nonexistantConnections[i]
        foundPair = False
        pairIndex = 0
        inNode = connection.getInputNode().getID()
        outNode = connection.getOutputNode().getID()
        while pairIndex < len(nonexistantPairs) and not foundPair:
            pair = nonexistantPairs[pairIndex]
            if pair[0] == inNode and pair[1] == outNode:
                foundPair = True
            else:
                pairIndex += 1
        if foundPair:
            del nonexistantPairs[pairIndex]
        else:
            success = False

    return success
    
def testNeuralNet():
    success = testNeuralNetConstructor()
    if success:
        print("Neural Net Constructor Test succeeded.")
    else:
        print("Neural Net Constructor Test failed.")
    success = testCreateNewNode()
    if success:
        print("Create New Node Test succeeded.")
    else:
        print("Create New Node Test failed.")
    success = testCreateNewInput()
    if success:
        print("Create New Input Test succeeded.")
    else:
        print("Create New Input Test failed.")
    success = testCreateNewOutput()
    if success:
        print("Create New Output Test succeeded.")
    else:
        print("Create New Output Test failed.")
    success = testInsertNewNode()
    if success:
        print("Insert New Node Test Test succeeded.")
    else:
        print("Insert New Node Test failed.")
    success = testGetNonexistantConnections()
    if success:
        print("Get Nonexistant Connections Test succeeded.")
    else:
        print("Get Nonexistant Connections Test failed.")

def fullTest():
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
    print("Testing NeuralNet.py")
    testNeuralNet()

main()
