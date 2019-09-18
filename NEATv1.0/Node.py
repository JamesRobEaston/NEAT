from Connection import Connection

class Node:

    def __init__(self, inConnections, outConnections, ID):
        self.value = 0
        self.inConnections = inConnections
        self.outConnections = outConnections
        self.ID = ID

    def addInConnection(self, newConnection):
        self.inConnections.append(newConnection)

    def removeInConnection(self, connection):
        self.inConnections.remove(connection)

    def addOutConnection(self, newConnection):
        self.outConnections.append(newConnection)

    def removeOutConnection(self, connection):
        self.outConnections.remove(connection)

    def getNumInConnections(self):
        return len(self.inConnections)

    def getNumOutConnections(self):
        return len(self.outConnections)

    def isInput(self):
        return len(self.inConnections) == 0

    def isOutput(self):
        return len(self.outConnections) == 0

    def getInConnections(self):
        return self.inConnections

    def getOutConnections(self):
        return self.outConnections

    def getOutNodes(self):
        outNodes = []
        for connection in self.outConnections:
            outNodes.append(connection.getOutputNode())
        return outNodes

    def getInNodes(self):
        inNodes = []
        for connection in self.inConnections:
            inNodes.append(connection.getInputNode())
        return inNodes

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

    def getID(self):
        return self.ID

    def calculateValue(self):
        value = 0
        for connection in self.inConnections:
            value += connection.getWeightedValue()
        self.setValue(value)
        return self.getValue()
