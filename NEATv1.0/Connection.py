class Connection:

    def __init__(self, weight, inNode, outNode, innovation):
        self.weight = weight
        self.inNode = inNode
        self.outNode = outNode
        self.enabled = True
        self.innovation = innovation
        inNode.addOutConnection(self)
        outNode.addInConnection(self)

    def disable(self):
        self.enabled = False
        self.inNode.removeOutConnection(self)
        self.outNode.removeInConnection(self)

    def enable(self):
        self.enabled = True
        self.outNode.addInConnection(self)
        self.inNode.addOutConnection(self)

    def getWeightedValue(self):
        return self.weight * inNode.getValue()

    def getWeight(self):
        return self.weight

    def setWeight(self, weight):
        self.weight = weight

    def getInputNode(self):
        return self.inNode

    def getOutputNode(self):
        return self.outNode

    def getInnovation(self):
        return self.innovation

    def setInnovation(self, innovation):
        self.innovation = innovation

    def isEnabled(self):
        return self.enabled
