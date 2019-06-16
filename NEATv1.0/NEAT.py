import time
from NeuralNet import NeuralNet

class NEAT:

    def __init__(self, generationSize, numInputs, numOutputs, eventEntities = None):
        self.networks = []
        self.innovationNumber = numInputs * numOutputs + 1
        self.nodeID = numInputs + numOutputs + 1
        self.newConnectionsThisGeneration = []
        self.newNodesThisGeneration = []
        self.currentSpeciation = []
        self.scoreTotals = []

        #Set various parameters
        self.compatibilityDistanceThreshold = 1.0
        self.existingConnectionMutationChance = 0.1
        self.nodeMutationChance = 0.3
        self.connectionMutationChance = 0.3
        self.requiredLongTermImprovement = 0.05
        self.numberOfGenerationsForLongTermImprovement = 20
        
        for i in range(generationSize):
            self.networks.append(NeuralNet(numInputs, numOutputs))

    def trainAndGetAllNetworksEveryRound(self, event):
        while event.continueTraining():
            generationScores = []
            for neuralNet in self.networks:
                generationScores.append(0)
            while not event.isComplete():
                for i in range(len(self.networks)):
                    neuralNet = self.networks[i]
                    score = event.doAction(neuralNet)
                    generationScores[i] = score
            self.networks = self.createNextGeneration(generationScores, self.networks)

    def trainAndGetActiveNetworksEveryRound(self, event):
        while event.continueTraining():
            generationScores = []
            for neuralNet in self.networks:
                generationScores.append(0)
            while not event.isComplete():
                for i in range(len(self.networks)):
                    if scores[i] == 0:
                        neuralNet = self.networks[i]
                        isFinished, score = event.doAction(neuralNet)
                        if isFinished:
                            generationScores[i] = score
            self.networks = self.createNextGeneration(generationScores, self.networks)

    def trainThroughEventEntities(self, event):
        while event.continueTraining():
            generationScores = []
            for neuralNet in self.networks:
                generationScores.append(0)
            eventEntities = event.getEntities()
            for i in range(len(eventEntities)):
                eventEntity.addNeuralNet(self.networks[i])
            while not event.isComplete():
                for i in range(len(eventEntities)):
                    entity = eventEntities[i]
                    score = entity.doAction()
                    generationScores[i] = score
            self.networks = self.createNextGeneration(generationScores, self.networks)
        
    def createNextGenerationFromNetworks(self, scores):
        return self.createNextGeneration(scores, self.networks)

    def createNextGeneration(self, scores, currentGeneration):
        #Perform speciation and calculate the adjusted scores
        speciation, speciationScores = self.speciate(self.currentSpeciation ,currentGeneration, scores, self.compatibilityDistanceThreshold)
        adjustedScores = self.calculateAdjustedFitnessScores(speciationScores)

        #Determine the number of offspring that each species gets
        allowedNumOffspring = []
        for species in speciation:
            allowedNumOffspring.append(0)
        
        #If the population has not increased by more than 5% in the number of generations for long term improvement, only allow the top two species to reproduce
        self.scoreTotals.insert(0, sum(scores))
        while len(self.scoreTotals) > self.numberOfGenerationsForLongTermImprovement:
            self.scoreTotals.pop(-1)
        isIncreasing = (len(self.scores) < self.numberOfGenerationsForLongTermImprovement) or determineIfFitnessIsIncreasing(self.totalScores, self.requiredImprovementOverTime)
        if not isIncreasing:
            #Find the top two species
            maxScore1 = sum(adjustedScores[1])
            maxIndex1 = 1
            maxScore2 = sum(adjustedScores[0])
            maxIndex2 = 0
            for i in range(len(adjustedScores)):
                if sum(adjustedScores[i]) >= maxScore1:
                    maxScore2 = maxScore1
                    maxIndex2 = maxIndex1
                    maxScore1 = adjustedScores[i]
                    maxIndex1 = i
            totalAdjustedScore = sum(adjustedScores[maxIndex1]) + sum(adjustedScores[maxIndex2])
            allowedNumOffspring[maxIndex1] = round(len(currentGeneration) * (adjustedScores1 / totalAdjustedScore))
            allowedNumOffspring[maxIndex2] = round(len(currentGeneration) * (adjustedScores2 / totalAdjustedScore))
            self.scoreTotals.clear()
        else:
            #Otherwise, allow each species an offspring population whose size is proportional to their contribution to the total adjusted score
            totalAdjustedScore = 0
            for adjustedSpeciesScores in adjustedScores:
                totalAdjustedScore += sum(adjustedSpeciesScores)
            for i in range(len(adjustedScores)):
                allowedNumOffspring[i] = round(sum(adjustedScores[i]) / totalAdjustedScore)
            speciesIndex = 0
            newPopulationSize = sum(allowedNumOffspring)
            while newPopulationSize < len(currentGeneration):
                allowedNumOffspring[speciesIndex] += 1
                speciesIndex += 1
                newPopulationSize += 1

        #Perform crossovers within the various species
        nextGeneration = []
        for i in range(len(speciation)):
            species = speciation[i]
            scores = adjustedScores[i]
            numOffspring = allowedNumOffspring[i]
            performCrossoverWithinSpecies(species, scores, nextGeneration, numOffspring)
        
        self.currentSpeciation = speciation
        self.newConnectionsThisGeneration = []
        self.newNodesThisGeneration = []
        return nextGeneration

    def performCrossoverWithinSpecies(self, species, scores, nextGeneration, numOffspring):
        #Eliminate the lower half of the species
        #Sort the species according to their score
        for i in range(len(scores) / 2):
            maxIndex = i
            for j in range(i, len(scores)):
                if scores[j] > scores[i]:
                    maxIndex = j
            prevScore = scores[i]
            prevNeuralNet = self.networks[i]
            scores[i] = scores[j]
            scores[j] = prevScore
            self.networks[i] = self.networks[j]
            self.networks[j] = self.networks[i]

        #Use the best half of the species to create offspring
        offspringProducers = []
        for i in range(len(scores)/2):
            offspringProducers.append(self.networks[i])
        producersSize = len(offspringProducers)

        random.seed(int(round(time.time() * 1000)))
        while len(nextGeneration) < numOffspring:
            neuralNet1Index = random.randRange(producersSize)
            neuralNet2Index = random.randRange(producersSize)
            canCrossover = neuralNet1Index != neuralNet2Index
            if canCrossover:
                newNetwork = self.performCrossover(offspringProducers[neuralNet1Index], offspringProducers[neuralNet2Index])
                self.mutate(newNetwork)
                nextGeneration.append(newNetwork)

    def calculateAdjustedFitnessScores(self, speciationScores):
        adjustedSpeciationScores = []
        for speciesScores in speciationScores:
            adjustedSpeciesScores = []
            speciesSize = len(speciesScores)
            for score in speciesScores:
                adjustedSpeciesScores.append((1.0 * score)/speciesSize)
            adjustedSpeciationScores.append(adjustedSpeciesScores)
        return adjustedSpeciationScores

    def speciate(self, prevSpeciation, generation, scores, compatibilityDistanceThreshold):
        random.seed(int(round(time.time() * 1000)))
        newSpeciation = []
        scoreSpeciation = []
        for species in self.allSpecies:
            newSpeciation.append([])
            scoreSpeciation.append([])
        for j in range(len(generation)):
            neuralNet = generation[j]
            speciesIndex = 0
            for i in range(len(self.allSpecies)):
                species = prevSpeciation[i]
                speciesRepresentative = prevSpeciation[random.randRange(len(prevSpeciation))]
                compatibilityDistance = self.calculateCompatibilityDistance(neuralNet, speciesRepresentative)
                speciesIndex += 1
                if compatibilityDistance < compatibilityDistanceThreshold:
                    newSpeciation[i].append(neuralNet)
                    scoreSpeciation[i].append(scores[j])
            if speciesIndex == (len(prevSpeciation) - 1):
                newSpecies = []
                newSpeciesScores = []
                newSpecies.append(neuralNet)
                newSpeciesScores.append(scores[j])
                newSpeciation.append(newSpecies)
                prevSpeciation.append(newSpecies)
                scoreSpeciation.append(newSpeciesScores)
        speciesIndex = 0
        while speciesIndex < len(newSpeciation):
            species = newSpeciation(speciesIndex)
            if len(species) == 0:
                newSpeciation.pop(speciesIndex)
            else:
                speciesIndex += 1
        return newSpeciation, scoreSpeciation

    def determineIfFitnessIsIncreasing(self, totalScores, requiredImprovement):
        isIncreasing = totalScores[0] > (totalScores[-1] * (1.0 + requiredImprovement))
        isIncreasing = isIncreasing or ((totalScores[0] > (totalScores[-2] * (1.0 + requiredImprovement))) and (totalScores[0] > (totalScores[-3] * (1.0 + requiredImprovement))))
        isIncreasing = isIncreasing or ((totalScores[0] > (totalScores[1] * (1.0 + requiredImprovement))) and (totalScores[0] > (totalScores[2] * (1.0 + requiredImprovement))))
        return isIncreasing

    def calculateCompatibilityDistance(self, neuralNet1, neuralNet2):
        excessCoefficient = 1.0
        disjointCoefficient = 1.0
        weightCoefficient = 0.1
        normalizingCoefficient = 1.0
        
        disjoint, excess = self.calculateDisjointAndExcess(neuralNet1, neuralNet2)
        averageWeightDiff = self.calculateAverageWeightDiff(neuralNet1, neuralNet2)
        numConnections1 = len(neuralNet1.getConnections())
        numConnections2 = len(neuralNet2.getConnections())
        if numConnections1 > numConnections2:
            normalizingCoefficient = numConnections1 * 1.0
        else:
            normalizingCoefficient = numConnections2 * 1.0

        if normalizingCoefficient < 20.0:
            normalizingCoefficient = 1.0

        compatibilityDistance = (excessCoefficient * excess) / normalizingCoefficient + (disjointCoefficient * disjoint) / normalizingCoefficient + weightCoefficient * averageWeightDiff
        return compatibilityDistance

    def calculateDisjointAndExcess(self, neuralNet1, neuralNet2):
        disjoint = 0
        excess = 0
        unsearchedConnections1 = neuralNet1.getConnections().copy()
        unsearchedConnections2 = neuralNet2.getConnections().copy()
        maxInnovation1 = unserachedConnections1[len(unsearchedConnections1)-1].getInnovation()
        maxInnovation2 = unserachedConnections2[len(unsearchedConnections2)-1].getInnovation()
        minInnovation = 0
        if maxInnovation1 < maxInnovation2:
            minInnovation = maxInnovation1
        else:
            minInnovation = maxInnovation2
        for connection1 in unsearchedConnections1:
            hasMatching = False
            connectionIndex = 0
            while (not hasMatching) and (connectionIndex < len(unsearchedConnections2)):
                connection2 = unsearchedConnections2[connectionIndex]
                hasMatching = connection1.getInnovation() == connection2.getInnovation()
                connectionIndex++
            if hasMatching:
                unsearchedConnections1.remove(connection1)
                unsearchedConnections2.remove(connection2)
            else:
                if connection1.getInnovation() < minInnovation:
                    disjoint += 1
                else:
                    excess += 1
                    
        for connection in unsearchedConnections2:
            if connection.getInnovation() < minInnovation:
                disjoint += 1
            else:
                excess += 1
        return disjoint, excess

    def calculateAverageWeightDiff(self, neuralNet1, neuralNet2):
        weightDifference = 0.0
        numMatching = 0.0
        unsearchedConnections1 = neuralNet1.getConnections().copy()
        unsearchedConnections2 = neuralNet2.getConnections().copy()
        for connection1 in unsearchedConnections1:
            hasMatching = False
            connectionIndex = 0
            while (not hasMatching) and (connectionIndex < len(unsearchedConnections2)):
                connection2 = unsearchedConnections2[connectionIndex]
                hasMatching = connection1.getInnovation() == connection2.getInnovation()
                connectionIndex++
            if hasMatching:
                unsearchedConnections1.remove(connection1)
                unsearchedConnections2.remove(connection2)
                weightDiff += abs(connection1.getWeight() - connection2.getWeight())
                numMatching += 1.0
        return weightDifference / numMatching
                

    def performCrossover(self, neuralNet1, neuralNet2):
        child = NeuralNet()
        
        insertedNodeIDs = []
        newNodes = []
        newConnections = []

        #Create the input and output layers for the new neural net
        inputs = neuralNet1.getInputs()
        for netInput in inputs:
            newNodes.append(child.createNewInput(netInput.getID()))
            insertedNodeIDs.append(netInput.getID())
        outputs = neuralNet1.getOutputs()
        for netOutput in outputs:
            newNodes.append(child.createNewOutput(netOutput.getID()))
            insertedNodeIDs.append(netOutput.getID())

        #Insert any other hidden nodes
        newNodes = neuralNet1.getNodes()
        for node in newNodes:
            if not self.contains(insertedNodeIDs, node.getID()):
                newNodes.append(child.createNewNode([], -1, node.getID()))
                insertedNodeIDs.append(node.getID())
        newNodes = neuralNet2.getNodes()
        for node in newNodes:
            if not self.contains(insertedNodeIDs, node.getID()):
                newNodes.append(child.createNewNode([], -1, node.getID()))
                insertedNodeIDs.append(node.getID())

        #Determine the connections for the new network
        newConnections = neuralNet1.getConnections()
        for connection in newConnections:
            newConnections.append([connection.getInNode().getID(), connection.getOutNode().getID(), connection.getWeight(), connection.getInnovation()])
        newConnections = neuralNet2.getConnections()
        for connection in newConnections:
            innovation = connection.getInnovation()
            #If both neural nets have the same connection, average their weights
            if (innovation <= len(newConnections)) and newConnections[innovation - 1][3] == innovation:
                newConnections[innovation-1][2] = (newConnections[innovation-1][2] + connection.getWeight())/2
            #Otherwise, insert the new connection into the list
            else:
                newConnections.insert(innovation - 1, [connection.getInNode().getID(), connection.getOutNode().getID(), connection.getWeight(), connection.getInnovation()])
        #Actually make the connections
        for connection in newConnections:
            inNode = self.findNode(newNodes, connection[0])
            outNode = self.findNode(newNodes, connection[1])
            newConnection = Connection(connection[2], inNode, outNode, connection[3])
            child.addConnection(newConnection)

        return child
            
    def findNode(self, array, nodeID):
        found = False
        node = None
        arrayIndex = 0
        while (not found) and (arrayIndex < len(array)):
            arrayItem = array[arrayIndex]
            found = arrayItem.getID() == nodeID
            if found:
                node = arrayItem
            arrayIndex += 1
        return node

    def contains(self, array, item):
        contains = False
        arrayIndex = 0
        while (not contains) and (arrayIndex < len(array)):
            arrayItem = array[arrayIndex]
            contains = arrayItem == item
            arrayIndex += 1
        return contains

    def mutate(self, neuralNet):
        self.existingConnectionMutation(self.existingConnectionMutationChance, neuralNet)
        self.nodeMutation(self.nodeMutationChance, neuralNet)
        self.connectionMutation(self.connectionMutationChance, neuralNet)
        
    def existingConnectionMutation(mutationChance, neuralNet):
        connections = neuralNet.getConnections()
        for connection in connections:
            random.seed(int(round(time.time() * 1000)))
            chance = random.randRange(0.0, 1.0, 0.01)
            if chance < mutationChance:
                connection.setWeight(random.randRange(-1.0, 1.0, 0.1))
            
    def nodeMutation(self, mutationChance, neuralNet, innovation):
        random.seed(int(round(time.time() * 1000)))
        chance = random.randRange(0.0, 1.0, 0.01)
        if chance < mutationChance:
            connections = neuralNet.getConnections()
            connection = connections[random.randint(0, len(connections) - 1)]
            innovation = self.innovationNumber
            nodeID = self.nodeID
            
            #Check if this innovation has occurred in this generation
            newInnovation = True
            nodeIndex = 0
            while newInnovation and nodeIndex < len(self.newNodesThisGeneration):
                nodeInnovation = self.newNodesThisGeneration[nodeIndex]
                inNodeID = nodeInnovation[0]
                outNodeID = nodeInnovation[1]
                if (connection.getInNode().getNodeID() == inNodeID) and (connection.getOutNode().getNodeID() == outNodeID):
                    newInnovation = False
                    nodeID = nodeInnovation[2]
                    innovation = nodeInnovation[3]

            #Add a new node with the appropriate values
            neuralNet.insertNewNode(connection, innovation, nodeID)

            #If this innovation is new, record it and update self.innovationNumber and self.nodeID
            if newInnovation:
                inNodeID = connection.getInNode().getID()
                outNodeID = connection.getOutNode().getID()
                self.newNodesThisGeneration.append([inNodeID, outNodeID, self.nodeID, self.innovationNumber])
                self.innovationNumber += 2
                self.nodeID += 1

    def connectionMutation(self, mutationChance, neuralNet, generation):
        random.seed(int(round(time.time() * 1000)))
        chance = random.randRange(0.0, 1.0, 0.01)
        if chance < mutationChance:
            possibleConnections = neuralNet.getNonExistantConnections(self.innovationNumber)
            if len(possibleConnections) > 0:
                connection = possibleConnections[random.randRange(len(possibleConnections))]
                
                #Check if this innovation has occurred in this generation
                newInnovation = True
                connectionIndex = 0
                while newInnovation and connectionIndex < len(self.newConnectionsThisGeneration):
                    connectionsInnovation = self.newConnectionsThisGeneration[connectionIndex]
                    inNodeID = connectionInnovation[0]
                    outNodeID = connectionInnovation[1]
                    if (connection.getInNode().getNodeID() == inNodeID) and (connection.getOutNode().getNodeID() == outNodeID):
                        newInnovation = False
                        connection.setInnovation(connectionInnovation[2])

                #Add the new connection
                neuralNet.addConnection(connection)
                
                #If this innovation is new, record it and update self.innovationNumber
                if newInnovation:
                    inNodeID = connection.getInNode().getID()
                    outNodeID = connection.getOutNode().getID()
                    self.newConnectionsThisGeneration.append([inNodeID, outNodeID, self.innovationNumber])
                    self.innovationNumber += 1

    def getNeuralNets(self):
        return self.networks

    def setNeuralNets(self, networks):
        self.networks = networks
