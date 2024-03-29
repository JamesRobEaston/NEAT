import random
import time
import matplotlib.pyplot as plt
from decimal import Context, Decimal, ROUND_HALF_UP
from Node import Node
from Connection import Connection

class NeuralNet:

    #The constructor for the Neural Net. Creates a neural net with the specified number of inputs and outputs where every
    #input is connected to every output. Each connection has a random weight.
    def __init__(self, numInputs = 0, numOutputs = 0):
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.connections = []
        random.seed()
        for i in range(numInputs):
            newInput = Node([], [], i + 1)
            self.inputs.append(newInput)
            self.nodes.append(newInput)

        for i in range(numOutputs):
            newOutput = Node([], [], numInputs + i + 1)
            for j in range(numInputs):
                connection = Connection((1.0 * random.randrange(-100, 100))/100.0, self.inputs[j], newOutput, i + j + 1)
                self.connections.append(connection)
            self.nodes.append(newOutput)
            self.outputs.append(newOutput)

    #The method to insert a new node into the neural net without inserting over a previous connection.
    #
    #Inputs: A list of nodes which are inputs for the new node
    #        The innovation number for the new node
    #        The new node's ID
    #
    #Outputs: The new node
    #
    #Side effects: The new node is added to this list of nodes and is inserted appropriately into the neural net
    def createNewNode(self, inNodes, innovation, nodeID):
        random.seed()
        newNode = Node([], [], nodeID)
        minIndex = len(self.inputs)
        for node in inNodes:
            index = self.nodes.index(node)
            if(index > minIndex):
                minIndex = index + 1
            connection = Connection((1.0 * random.randrange(-100, 100))/100.0, node, newNode, innovation)
            self.connections.append(connection)
            
        self.nodes.insert(minIndex, newNode)
        return newNode

    #The method to create a new node that is an input
    #
    #Inputs: The node's ID
    #
    #Outputs: The new input node
    #
    #Side effects: The new input is added to this neural net's list of inputs and nodes
    def createNewInput(self, nodeID):
        random.seed()
        newNode = Node([], [], nodeID)
        self.nodes.insert(0, newNode)
        self.inputs.append(newNode)
        return newNode

    #The method to create a new node that is an output
    #
    #Inputs: The node's ID
    #
    #Outputs: The new output node
    #
    #Side effects: The new output is added to this neural net's list of output and nodes
    def createNewOutput(self, nodeID):
        random.seed()
        newNode = Node([], [], nodeID)
        self.nodes.append(newNode)
        self.outputs.append(newNode)
        return newNode

    #The method to insert a new node into the neural net by "breaking" an existing connection. That is, a new node
    #is created such that it has a connection to the given connection's input and a connection to the given connection's
    #output.
    #
    #Inputs: The connection to be broken
    #        The innovation number for the new node
    #        The new node's ID
    #
    #Outputs: The new node
    #
    #Side effects: The new node is added to this list of nodes and is inserted appropriately into the neural net.
    #              The connection given is disabled.
    def insertNewNode(self, connection, innovation, nodeID):
        newNode = Node([], [], nodeID)
        index = self.nodes.index(connection.getInputNode()) + 1
        if len(self.inputs) > index:
            index = len(self.inputs)
        newConnection1 = Connection(1.0, connection.getInputNode(), newNode, innovation)
        newConnection2 = Connection(connection.getWeight(), newNode, connection.getOutputNode(), innovation + 1)
        self.nodes.insert(index, newNode)
        self.connections.append(newConnection1) 
        self.connections.append(newConnection2)
        connection.disable()
        return newNode

    #The method to disable a connection.
    #
    #Inputs: The node which is the input for the connection
    #        The node which is the output for the connection
    #
    #Outputs: None
    #
    #Side effects: The connection specified by the inputs is disabled.
    def disableConnection(self, inNode, outNode):
        for connection in self.connections:
            if connection.isEnabled() and connection.getInputNode() == inNode and connection.getOutputNode() == outNode:
                connection.disable()

    #The method to get all of the connections between two nodes which do not currently have connections. These connections are disabled
    #by default.
    #
    #Inputs: None
    #
    #Outputs: A list of diabled connections which link currently unlinked nodes.
    #
    #Side effects: None
    def getNonexistantConnections(self):
        random.seed()
        nonExistantConnections = []
        for node in self.nodes:
            #Output nodes and nodes that are already connected to all other valid nodes do not have any possible out connections
            isOutputNode = node.getNumOutConnections() == 0
            if not isOutputNode:
                validOutputs = []
                for validNode in self.nodes:
                    validOutputs.append(validNode)

                #Remove all nodes before this node
                while validOutputs[0] != node:
                    validOutputs.pop(0)
                
                #Remove this node since it cant connect to itself
                validOutputs.pop(0)

                #Remove all input nodes
                while validOutputs[0].getNumInConnections() == 0:
                    validOutputs.pop(0)

                #Remove all nodes that this node is already connected to
                existingOutputs = node.getOutNodes()
                for output in existingOutputs:
                    if not self.contains(validOutputs, output):
                        print(node.getID())
                        print(output.getID())
                        print("--")
                        for out in output.outConnections:
                            print(out.getOutputNode().getID())
                        print("--")
                        for out in existingOutputs:
                            print(out.getID())
                        print("--")
                        for node in self.nodes:
                            print(node.getID())
                    validOutputs.remove(output)

                for output in validOutputs:
                    connection = Connection((1.0 * random.randrange(-100, 100))/100.0, node, output, innovation)
                    connection.disable()
                    nonExistantConnections.append(connection)
       
        return nonExistantConnections

    #A helper method to determine if an array contains an element
    #
    #Inputs: The array and element
    #
    #Outputs: A boolean indicating if the element is in the array
    #
    #Side effects: None
    def contains(self, array, element):
        contains = False
        arrayIndex = 0
        while (not contains) and (arrayIndex < len(array)):
            contains = array[arrayIndex] == element
            arrayIndex += 1
    
        return contains

    #The method to add a connection to this neural net
    #
    #Inputs: The connection to be added
    #
    #Outputs: None
    #
    #Side effects: The given connection is added to the list of connections.
    def addConnection(self, connection):
        if not connection.isEnabled():
            connection.enable()
        outputNode = connection.getOutputNode()
        inputNode = connection.getInputNode()
        inputIndex = self.nodes.index(inputNode)
        outputIndex = self.nodes.index(outputNode)
        if inputIndex > outputIndex:
            self.nodes.remove(outputNode)
            self.nodes.insert(inputIndex, outputNode)
        self.connections.insert(connection.getInnovation()-1, connection)

    #The method to compute the final values of the output layer given values for the input layer.
    #
    #Inputs: A list of the input values
    #
    #Outputs: A list of values which are the values of the output nodes in the order of the nodes.
    #
    #Side effects: The values of the nodes will be changed appropriately.  
    def feedforward(self, inputs):
        for i in range(len(inputs)):
            self.inputs[i].setValue(inputs[i])
        for node in self.nodes:
            node.calculateValue()
        values = []
        for node in self.outputs:
            values.append(node.getValue)
        return values

    #The method to get a connection by its innovation number
    #
    #Inputs: The innovation number of the connection
    #
    #Outputs: The connection with the given innovation number
    #
    #Side effects: None
    def getConnectionByInnovationNumber(self, innovationNumber):
        connection = None
        connectionIndex = 0
        while connection == None and connectionIndex < len(self.connections):
            possibleConnection = self.connections[connectionIndex]
            if possibleConnections.innocatoinNumber == innocationNumber:
               connection = possibleConnection
        return connection

    def getConnections(self):
        return self.connections

    def getEnabledConnections(self):
        enabledConnections = []
        for connection in self.connections:
            if connection.isEnabled():
                enabledConnections.append(connection)
        return enabledConnections

    def getNodes(self):
        return self.nodes

    def getInputs(self):
        return self.inputs

    def getOutputs(self):
        return self.outputs

    def getNodeIndex(self, nodeID):
        nodeIndex = -1
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if nodeID == node.getID():
                nodeIndex = i
        return nodeIndex

    #A helper method to determine if the nodes of this neural net are ordered in so that the
    #outputs for any node after after that node
    #
    #Inputs: None
    #
    #Outputs: Whether the nodes in this neural network are ordered
    #
    #Side effects: None
    def isOrdered(self):
        isOrdered = True
        nodeIndex = 0
        offNodeIn = -1
        offNodeOut = -1
        while isOrdered and nodeIndex < len(self.nodes):
            node = self.nodes[nodeIndex]
            outNodes = node.getOutNodes()
            thisIndex = self.getNodeIndex(node.getID())
            outNodeIndex = 0
            while isOrdered and outNodeIndex < len(outNodes):
                outNode = outNodes[outNodeIndex]
                outNodeListIndex = self.getNodeIndex(outNode.getID())
                isOrdered = isOrdered and thisIndex < outNodeListIndex
                outNodeIndex += 1
            nodeIndex += 1
        if not isOrdered:
            offNodeIn = node.getID()
            offNodeOut = outNode.getID()
        return isOrdered, offNodeIn, offNodeOut

    #All of the methods below this point are helper methods to display this neural network.
    def show(self):
        xValues = []
        yValues = []
        for node in self.nodes:
            xValues.append(0)
            yValues.append(0)
            
        for node in self.inputs:
            self.recursiveDFS(node, 1, xValues)
            
        maxOutput = 0
        lowestOutputIndex = self.getNodeIndex(self.outputs[0].getID())
        for node in self.outputs:
            outputIndex = self.getNodeIndex(node.getID())
            if xValues[outputIndex] > maxOutput:
                maxOutput = xValues[outputIndex]
            if outputIndex < lowestOutputIndex:
                lowestOutputIndex = outputIndex
        outputIndex = lowestOutputIndex
        while outputIndex < len(xValues):
            xValues[outputIndex] = maxOutput
            outputIndex += 1
        
        maxNumNodesAtALocation = 0
        nodeIndicesAtXLocations = []
        for i in range(maxOutput):
            numNodesAtXLocation = 0
            nodeIndicesAtXLocation = []
            for j in range(len(xValues)):
                if xValues[j] == (i + 1):
                    numNodesAtXLocation += 1
                    nodeIndicesAtXLocation.append(j)
            if numNodesAtXLocation > maxNumNodesAtALocation:
                maxNumNodesAtALocation = numNodesAtXLocation
            nodeIndicesAtXLocations.append(nodeIndicesAtXLocation)
        for nodeIndices in nodeIndicesAtXLocations:
            for i in range(len(nodeIndices)):
                index = nodeIndices[i]
                yValues[index] = 1.0 + (1.0 * i) + (1.0*(maxNumNodesAtALocation - len(nodeIndices)))/2
        
        lines = []
        for node in self.inputs:
            lineX = []
            lineY = []
            nodeIndex = self.getNodeIndex(node.getID())
            lineX.append(xValues[nodeIndex])
            lineY.append(yValues[nodeIndex])
            self.recursivelyCreateLines(node, lines, [], [], xValues, yValues)

        for line in lines:
            plt.plot(line[0], line[1], marker='o', markerfacecolor='red', markersize=6, color='black', linewidth=2)
        plt.show()

    def recursivelyCreateLines(self, node, lines, lineX, lineY, xLocations, yLocations):
        nodesToAdd = node.getOutNodes()
        nodeIndex = self.getNodeIndex(node.getID())
        lineX.append(xLocations[nodeIndex])
        lineY.append(yLocations[nodeIndex])
        if len(nodesToAdd) == 0:
            line = []
            line.append(lineX)
            line.append(lineY)
            lines.append(line)
        else:
            for newNode in nodesToAdd:
                self.recursivelyCreateLines(newNode, lines, lineX.copy(), lineY.copy(), xLocations, yLocations)   

    def recursiveDFS(self, node, nodeDepth, nodeDepths):
        nodesToExplore = node.getOutNodes()
        if nodeDepth > nodeDepths[self.getNodeIndex(node.getID())]:
            nodeDepths[self.getNodeIndex(node.getID())] = nodeDepth
        for nextNode in nodesToExplore:
            self.recursiveDFS(nextNode, nodeDepth+1, nodeDepths)
                
    def round(self, num):
        return int(Context(rounding=ROUND_HALF_UP).to_integral_exact(Decimal(num)))
