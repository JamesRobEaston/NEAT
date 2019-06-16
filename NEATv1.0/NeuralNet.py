import random
import time
import matplotlib.pyplot as plt
from Node import Node
from Connection import Connection

class NeuralNet:

    def __init__(self, numInputs = 0, numOutputs = 0):
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.connections = []
        random.seed(int(round(time.time() * 1000)))
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

    def createNewNode(self, inNodes, innovation, nodeID):
        random.seed(int(round(time.time() * 1000)))
        newNode = Node([], [], nodeID)
        maxIndex = 0
        for node in inNodes:
            index = self.nodes.index(node)
            if(index > maxIndex):
                maxIndex = index
            connection = Connection((1.0 * random.randrange(-100, 100))/100.0, node, newNode, innovation)
            self.connections.append(connection)
            
        self.nodes.insert(newNode, maxIndex+1)
        return newNode

    def createNewInput(self, nodeID):
        newNode = self.createNewNode([], -1, nodeID)
        self.inputs.append(newNode)
        return newNode

    def createNewOutput(self, nodeID):
        newNode = self.createNewNode([], -1, nodeID)
        self.inputs.append(newNode)
        return newNode

    def insertNewNode(self, connection, innovation, nodeID):
        newNode = Node([], [], nodeID)
        index = self.nodes.index(connection.getInputNode()) + 1
        newConnection1 = Connection(1.0, connection.getInputNode(), newNode, innovation)
        newConnection2 = Connection(connection.getWeight(), newNode, connection.getOutputNode(), innovation + 1)
        self.nodes.insert(index, newNode)
        self.connections.append(newConnection1) 
        self.connections.append(newConnection2)
        connection.disable()

    def disableConnection(self, inNode, outNode):
        for connection in self.connections:
            if connection.isEnabled() and connection.getInputNode == inNode and connection.getOutputNode() == outNode:
                connection.disable()

    def getNonexistantConnections(self, innovation):
        random.seed(int(round(time.time() * 1000)))
        nonExistantConnections = []
        for node in self.nodes:
            #Output nodes and nodes that are already connected to all other valid nodes do not have any possible connections
            isOutputNode = node.getNumOutConnections() == 0
            if not isOutputNode:
                validOutputs = self.nodes.copy()
                #Remove all nodes before this node
                while validOutputs[0] != node:
                    validOutputs.pop(0)
                #Remove all input nodes from the valid nodes since they cannot be connected to
                while validOutputs[0].getNumInConnections() == 0:
                    validOutputs.pop(0)
                #Remove all nodes that this node is already connected to
                existingOutputs = node.getOutNodes()
                for output in existingOutputs:
                    validOutputs.remove(output)
                    
                for output in validOutputs:
                    connection = Connection((1.0 * random.randrange(-100, 100))/100.0, node, output, innovation)
                    connection.disable()
                    nonExistantConnections.append(connection)
        return nonExistantConnections

    def addConnection(self, connection):
        if not connection.isEnabled():
            connection.enable()
        outputNode = connection.getOutNode()
        inputNode = connection.getInNode()
        inputIndex = self.nodes.index(inputNode)
        outputIndex = self.nodes.index(outputNode)
        if inputIndex > outputIndex:
            self.nodes.remove(outputNode)
            self.nodes.insert(inputIndex, outputNode)
        self.connections.insert(connection.getInnovation()-1, connection)
                     
    def feedforward(self, inputs):
        for i in range(len(inputs)):
            self.inputs[i].setValue(inputs[i])
        for node in self.nodes:
            node.calculateValue()
        values = []
        for node in self.outputs:
            values.append(node.getValue)
        return values

    def getConnections(self):
        return self.connections

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
                yValues[index] = 1.0 + (1.0 * i) * ((1.0 * maxNumNodesAtALocation)) / len(nodeIndices)
        
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
                
