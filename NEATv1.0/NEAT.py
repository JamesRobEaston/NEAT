import random
from decimal import Context, Decimal, ROUND_HALF_UP
from NeuralNet import NeuralNet
from Connection import Connection

#This class handles managing and performing the NEAT algorithm for other applications.
class NEAT:

    def __init__(self, generationSize = 100, numInputs = 0, numOutputs = 0, eventEntities = None, compatibilityDistanceThreshold = 1.0, existingConnectionMutationChance = 0.1, nodeMutationChance = 0.3, newConnectionMutationChance = 0.3, requiredLongTermImprovement = 0.05, numberOfGenerationsForLongTermImprovement = 20):
        #Initialize the various lists needed to perform NEAT
        self.networks = []
        self.innovationNumber = numInputs * numOutputs + 1
        self.nodeID = numInputs + numOutputs + 1
        self.newConnectionsThisGeneration = []
        self.newNodesThisGeneration = []
        self.currentSpeciation = []
        self.scoreTotals = []

        #Set various parameters given by the user
        self.compatibilityDistanceThreshold = compatibilityDistanceThreshold
        self.existingConnectionMutationChance = existingConnectionMutationChance
        self.nodeMutationChance = nodeMutationChance
        self.connectionMutationChance = newConnectionMutationChance
        self.requiredLongTermImprovement = requiredLongTermImprovement
        self.numberOfGenerationsForLongTermImprovement = numberOfGenerationsForLongTermImprovement

        #Create the NeuralNets
        for i in range(generationSize):
            self.networks.append(NeuralNet(numInputs, numOutputs))

    #The first method to allow users to train through this NEAT object.
    #This method is meant to allow a user to create an event and continually use and improve the neural networks to complete some action.
    #This method will continually run while the event object determines it is appropriate to continue training. While it is appropriate to
    #continue training, this method will go through "rounds" where the event object determines when a round is finished. For each round, this
    #method passes each neural network to the event object to be used in some way (most likely to generate a decision).
    #Once the round ends, the algorithm will use the NEAT algorithm to improve the neural nets and will move into the next round if appropriate.
    #
    #The primary factor that differentiates this method from the following method is that it passes every neural net through to the event object
    #each round, whereas the following method uses the event object to determine when neural nets are finished for the round, in which case they
    #are no longer passed to the event object. This method is meant to make it so users can track which neural net they are getting with more ease.
    #
    #Inputs: An event object, which must have three methods:
    #           continueTraining(): the method to determine when training should end. This method is called after each "round" is complete.
    #           isRoundComplete(): the method to determine when a round is complete. This is called after all of the neural neworks have been
    #                              passed to the event for a given round.
    #           doAction(neuralNet): the method that allows the event object to use neural nets in each round. This should take a neural network
    #                                as a parameter and return an integer score indicating how well the network did. Note that this score is recorded
    #                                each time this method is called, and so the only time an accurate score truly needs to be returned is on the
    #                                final round.
    def trainAndGetAllNetworksEveryRound(self, event):
        while event.continueTraining():
            #Initialize the scores for this generation
            generationScores = []
            for neuralNet in self.networks:
                generationScores.append(0)
            #Go through the round
            while not event.isRoundComplete():
                #Pass all of ther nerual networks to the event and record the returned score
                for i in range(len(self.networks)):
                    neuralNet = self.networks[i]
                    score = event.doAction(neuralNet)
                    generationScores[i] = score
            self.networks = self.createNextGeneration(generationScores, self.networks)

    #The second method to allow users to train through this NEAT object.
    #This method is meant to allow a user to create an event and continually use and improve the neural networks to complete some action.
    #This method will continually run while the event object determines it is appropriate to continue training. While it is appropriate to
    #continue training, this method will go through "rounds" where the event object determines when a round is finished. For each round, this
    #method passes the neural network which have not had a score returned to the event object to be used in some way (most likely to generate
    #a decision).
    #Once the round ends, the algorithm will use the NEAT algorithm to improve the neural nets and will move into the next round if appropriate.
    #
    #The primary factor that differentiates this method from the previous method is that it passes the neural nets that do not have a score through
    #to the event object each round, whereas the previous method passes every neural net to the event object each round. This method is meant to
    #make it so users can only get active neural nets each round
    #
    #Inputs: An event object, which must have three methods:
    #           continueTraining(): the method to determine when training should end. This method is called after each "round" is complete.
    #           isRoundComplete(): the method to determine when a round is complete. This is called after all of the neural neworks have been
    #                              passed to the event for a given round.
    #           doAction(neuralNet): the method that allows the event object to use neural nets in each round. This should take a neural network
    #                                as a parameter and return an integer score indicating how well the network did. Note that this score is recorded
    #                                each time this method is called, and so the only time an accurate score truly needs to be returned is on the
    #                                final round.
    def trainAndGetActiveNetworksEveryRound(self, event):
        while event.continueTraining():
            #Initialize the scores for this generation
            generationScores = []
            for neuralNet in self.networks:
                generationScores.append(0)
            #Go through the round
            while not event.isRoundComplete():
                #Pass the non-finished neural networks to the event
                for i in range(len(self.networks)):
                    if scores[i] == 0:
                        neuralNet = self.networks[i]
                        isFinished, score = event.doAction(neuralNet)
                        if isFinished:
                            generationScores[i] = score
            self.networks = self.createNextGeneration(generationScores, self.networks)

    #The third method to allow users to interact with the NEAT algorithm.
    #This method allows users to assign neural networks to an object which will then be called continually to perform actions.
    #
    #This method will continually run while the event object determines it is appropriate to continue training. While it is appropriate to
    #continue training, this method will go through "rounds" where the event object determines when a round is finished. Prior to each round,
    #the method gets a list of object which are given a neural network and can interact with the neural net when they are called to perform
    #an action. Once this  list is gotten, each entity is assigned a neural net (the first n entities will be assigned to the first n neural
    #nets and it is assumed that the number of entities does not exceed the number of neural nets). Once the round starts, each entity will be
    #called on to perform an action and the score gotten from this action is recorded by the algorithm.
    #Once the round ends, the algorithm will use the NEAT algorithm to improve the neural nets and will move into the next round if appropriate.
    #
    #Inputs: An event object which controls when training and the rounds end and that can be used to get the entities which use the neural nets.
    #        This object must have three methods:
    #           continueTraining(): the method to determine when training should end. This method is called after each "round" is complete.
    #           isRoundComplete(): the method to determine when a round is complete. This is called after all of the neural neworks have been
    #                              passed to the event for a given round.
    #           getEntities(): the method to get the entites to be used in training. These entities should have two methods:
    #                           addNeuralNet(neuralNet): the method to allow the entity to get a new neural net
    #                           doAction(): the method to prompt the entity to perform an action. This method should return a score indicating
    #                                       the performance of the entity (and indirectly the performance of the neural net).
    def trainThroughEventEntities(self, event):
        while event.continueTraining():
            generationScores = []
            for neuralNet in self.networks:
                generationScores.append(0)
            eventEntities = event.getEntities()
            for i in range(len(eventEntities)):
                eventEntity.addNeuralNet(self.networks[i])
            while not event.isRoundComplete():
                for i in range(len(eventEntities)):
                    entity = eventEntities[i]
                    score = entity.doAction()
                    generationScores[i] = score
            self.networks = self.createNextGeneration(generationScores, self.networks)

    #A getter for the neural nets used in the NEAT algorithm. This method can be used if the users of this library does not want to use one
    #of the above methods to handle training and instead wants to get the neural nets, perform training, and then allow this NEAT object to
    #perform the NEAT algorithm.
    def getNeuralNets(self):
        return self.networks

    #The method to create a new generation of neural networks using the neural nets stored in self.networks as the previous generation
    #
    #Inputs: scores - a list of scores of each neural net in self.networks. It is assumed that scores[i] is the score for self.networks[i].
    #
    #Outputs: A list of neural networks which is the next generation.
    def createNextGenerationFromNetworks(self, scores):
        return self.createNextGeneration(scores, self.networks)

    #The method to handle creating the next generation
    #
    #Inputs: scores - A list of scores for the current generation where scores[i] is the score for currentGeneration[i]
    #        currentGeneration - A list of neural nets which are to be used to create the next generation
    #
    #Outputs: A list of neural nets which is the next generation
    def createNextGeneration(self, scores, currentGeneration):
        print("Creating Next Generation")
        #Perform speciation and calculate the adjusted scores
        speciation, speciationScores = self.speciate(self.currentSpeciation, currentGeneration, scores, self.compatibilityDistanceThreshold)
        adjustedScores = self.calculateAdjustedFitnessScores(speciationScores)

        #Determine the number of offspring that each species gets
        allowedNumOffspring = []
        for species in speciation:
            allowedNumOffspring.append(0)
        
        #If the population has not increased by more than the required amount in the number of generations for long term improvement, only allow
        #the top two species to reproduce
        self.scoreTotals.insert(0, sum(scores))
        while len(self.scoreTotals) > self.numberOfGenerationsForLongTermImprovement:
            self.scoreTotals.pop(-1)
        isIncreasing = (len(self.scoreTotals) < self.numberOfGenerationsForLongTermImprovement) or determineIfFitnessIsIncreasing(self.totalScores, self.requiredImprovementOverTime)
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
            allowedNumOffspring[maxIndex1] = self.round(len(currentGeneration) * (adjustedScores1 / totalAdjustedScore))
            allowedNumOffspring[maxIndex2] = self.round(len(currentGeneration) * (adjustedScores2 / totalAdjustedScore))
            self.scoreTotals.clear()
        else:
            #Otherwise, allow each species an offspring population whose size is proportional to their contribution to the total adjusted score
            totalAdjustedScore = 0
            for adjustedSpeciesScores in adjustedScores:
                totalAdjustedScore += sum(adjustedSpeciesScores)
            for i in range(len(adjustedScores)):
                allowedNumOffspring[i] = self.round((sum(adjustedScores[i]) / totalAdjustedScore))
            speciesIndex = 0
            newPopulationSize = sum(allowedNumOffspring)
            while newPopulationSize < len(currentGeneration):
                allowedNumOffspring[speciesIndex] += 1
                speciesIndex = (speciesIndex + 1) % len(allowedNumOffspring)
                newPopulationSize = newPopulationSize + 1

        #Perform crossovers within the various species
        nextGeneration = []
        for i in range(len(speciation)):
            species = speciation[i]
            scores = adjustedScores[i]
            numOffspring = allowedNumOffspring[i]
            self.performCrossoverWithinSpecies(species, scores, nextGeneration, numOffspring)
        
        self.currentSpeciation = speciation
        self.newConnectionsThisGeneration = []
        self.newNodesThisGeneration = []
        print("Finished Next Generation")
        return nextGeneration

    #The method to handle performing all of the crossovers necessary for a given species. This essentially creates the next generation for this species.
    #
    #Inputs: species - A list of neural networks which is the species
    #        scores - A list of scores for this species where scores[i] is the score for species[i]
    #        nextGeneration - A list to store the next generation for this species
    #        numOffspring - The number of offspring allowed for this species.
    def performCrossoverWithinSpecies(self, species, scores, nextGeneration, numOffspring):
        #Eliminate the lower half of the species
        #Sort the species according to their score
        print("Entering Crossover")
        for i in range(self.round(1.0 * len(scores) / 2)):
            maxIndex = i
            for j in range(i, len(scores)):
                if scores[j] > scores[i]:
                    maxIndex = j
            prevScore = scores[i]
            prevNeuralNet = species[i]
            scores[i] = scores[j]
            scores[j] = prevScore
            species[i] = species[j]
            species[j] = species[i]

        #Use the best half of the species to create offspring
        offspringProducers = []
        for i in range(self.round(1.0 * len(scores)/2)):
            offspringProducers.append(species[i])
        producersSize = len(offspringProducers)
        
        random.seed()
        numProduced = 0
        while numProduced < numOffspring:
            neuralNet1Index = random.randrange(producersSize)
            neuralNet2Index = random.randrange(producersSize)
            newNetwork = self.performCrossover(offspringProducers[neuralNet1Index], offspringProducers[neuralNet2Index])
            self.mutate(newNetwork)
            nextGeneration.append(newNetwork)
            numProduced += 1
        print("Exiting Crossover")

    #The method to calculate the adjusted fitness score for a set of species. This essentially averages every score within a species
    #by the size of its species.
    #
    #Inputs: speciationScores - A list of lists of scores which are the scores for each neuralNet in each species. The inner lists are
    #                           organized so that they are separated by species.
    #
    #Outputs: A list of lists of scores which are the adjusted scores for each species.
    def calculateAdjustedFitnessScores(self, speciationScores):
        adjustedSpeciationScores = []
        for speciesScores in speciationScores:
            adjustedSpeciesScores = []
            speciesSize = len(speciesScores)
            for score in speciesScores:
                adjustedSpeciesScores.append((1.0 * score)/speciesSize)
            adjustedSpeciationScores.append(adjustedSpeciesScores)
        return adjustedSpeciationScores

    #This method handles organzing a generation into a list of species.
    #
    #Inputs: prevSpeciation - The previous list of lists of neural nets which was the previous list of species
    #        generation - The generation of neural nets to be organized by species
    #        scores - The scores for the neural nets in the generation
    #        compatibilityDistanceThreshold - The threshold for the how different two neural nets can be and still be considered the
    #                                         species (this distance is the compatibility distance)
    #
    #Outputs: A list of lists of neural nets which is the organization of all of the species,
    #         The scores of the neural nets so that newScores[i][j] is the score for newSpeciation[i][j]
    def speciate(self, prevSpeciation, generation, scores, compatibilityDistanceThreshold):
        random.seed()
        newSpeciation = []
        scoreSpeciation = []
        #Retain all of the old species
        for species in prevSpeciation:
            newSpeciation.append([])
            scoreSpeciation.append([])
        #Organize the generation into its list of species
        for j in range(len(generation)):
            neuralNet = generation[j]
            speciesIndex = 0
            #Determine which species this neural net is in
            for i in range(len(prevSpeciation)):
                species = prevSpeciation[i]
                speciesRepresentative = species[random.randrange(len(species))] #Use a random neural net from the species as a representative
                speciesIndex += 1
                compatibilityDistance = self.calculateCompatibilityDistance(neuralNet, speciesRepresentative) #Calculate how different this neural net
                                                                                                              #is from the representative
                #If the neural net is close enough to this representative, it is part of the species
                if compatibilityDistance < compatibilityDistanceThreshold:
                    newSpeciation[i].append(neuralNet)
                    scoreSpeciation[i].append(scores[j])
            #If the neural net did not find a species which it was close enough to, it is a new species
            if speciesIndex == (len(prevSpeciation)):
                newSpecies = []
                newSpeciesScores = []
                newSpecies.append(neuralNet)
                newSpeciesScores.append(scores[j])
                newSpeciation.append(newSpecies)
                prevSpeciation.append(newSpecies)
                scoreSpeciation.append(newSpeciesScores)
        #Loop through our new list of species and remove all empty species (aka all old species which have no new members)
        speciesIndex = 0
        while speciesIndex < len(newSpeciation):
            species = newSpeciation[speciesIndex]
            if len(species) == 0:
                newSpeciation.pop(speciesIndex)
                scoreSpeciation.pop(speciesIndex)
            else:
                speciesIndex += 1
                
        return newSpeciation, scoreSpeciation

    #A helper method to determine if the fitness of the generations has been increasing by a given amount
    #
    #Inputs: totalScores - A list of all of the scores from various generations
    #        requiredImprovement - The precent amount of improvement required for the scores to be considered increasing (in decimal form; i.e. 0.8 is 80%)
    #
    #Outputs: A boolean indicating if the scores are increasing
    def determineIfFitnessIsIncreasing(self, totalScores, requiredImprovement):
        isIncreasing = totalScores[0] > (totalScores[-1] * (1.0 + requiredImprovement))
        isIncreasing = isIncreasing or ((totalScores[0] > (totalScores[-2] * (1.0 + requiredImprovement))) and (totalScores[0] > (totalScores[-3] * (1.0 + requiredImprovement))))
        isIncreasing = isIncreasing or ((totalScores[0] > (totalScores[1] * (1.0 + requiredImprovement))) and (totalScores[0] > (totalScores[2] * (1.0 + requiredImprovement))))
        return isIncreasing

    #A helper method to calculate the compatibility distance between two neural nets
    #
    #Inputs: Two neural nets which need to have their compatibility distance computed
    #
    #Outputs: The compatibility distanve of the two neural nets
    def calculateCompatibilityDistance(self, neuralNet1, neuralNet2):
        excessCoefficient = 1.0
        disjointCoefficient = 1.0
        weightCoefficient = 0.1
        normalizingCoefficient = 1.0

        #Calculate the compatibility distance, as layed out in the NEAT paper
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

    #A helper method to determine the disjoint and excess between two neural nets, as detailed in the NEAT paper
    #
    #Inputs: The two neural nets which should have their disjoint and excess calculated
    #
    #Outputs: The disjoint and excess of the neural nets, in that order.
    def calculateDisjointAndExcess(self, neuralNet1, neuralNet2):
        disjoint = 0
        excess = 0
        unsearchedConnections1 = neuralNet1.getConnections().copy()
        unsearchedConnections2 = neuralNet2.getConnections().copy()
        maxInnovation1 = unsearchedConnections1[len(unsearchedConnections1)-1].getInnovation()
        maxInnovation2 = unsearchedConnections2[len(unsearchedConnections2)-1].getInnovation()
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
                connectionIndex += 1
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

    #A helper method to determine the average difference in shared weights between two neural networks
    #
    #Inputs: The two neural nets which should have their average weight difference calculated
    #
    #Outputs: The average weight difference of the two neural nets
    def calculateAverageWeightDiff(self, neuralNet1, neuralNet2):
        weightDifference = 0.0
        numMatching = 0.0
        unsearchedConnections1 = neuralNet1.getConnections().copy()
        unsearchedConnections2 = neuralNet2.getConnections().copy()
        #Search through the connections of the first neural net
        for connection1 in unsearchedConnections1:
            hasMatching = False
            connectionIndex = 0
            #See if the second neural net has a matching connection
            while (not hasMatching) and (connectionIndex < len(unsearchedConnections2)):
                connection2 = unsearchedConnections2[connectionIndex]
                #The innovation number acts effectively as an ID for connections among neural nets so we search for a matching
                #innovation number
                hasMatching = connection1.getInnovation() == connection2.getInnovation()
                connectionIndex += 1
            #If a matching weight is found, add it to the total weight difference and record another match
            if hasMatching:
                unsearchedConnections1.remove(connection1)
                unsearchedConnections2.remove(connection2)
                weightDifference += abs(connection1.getWeight() - connection2.getWeight())
                numMatching += 1.0
        return weightDifference / numMatching
                
    #A helper method to perform crossover between two networks
    #
    #Inputs: The two neural nets to be crossed over
    #
    #Outputs: A neural net which is the crossover of the two given neural nets
    def performCrossover(self, neuralNet1, neuralNet2):
        child = NeuralNet()
        
        insertedNodeIDs = []
        newNodes = []
        newConnections = []

        #Create the input and output layers for the new neural net
        inputs = neuralNet1.getInputs()
        for netInput in inputs:
            newNode = child.createNewInput(netInput.getID())
            newNodes.append(newNode)
            insertedNodeIDs.append(netInput.getID())
        outputs = neuralNet1.getOutputs()
        for netOutput in outputs:
            newNodes.append(child.createNewOutput(netOutput.getID()))
            insertedNodeIDs.append(netOutput.getID())
            
        #Insert any other hidden nodes
        potentialNewNodes = neuralNet1.getNodes()
        for node in potentialNewNodes:
            if not self.contains(insertedNodeIDs, node.getID()):
                newNodes.append(child.createNewNode([], -1, node.getID()))
                insertedNodeIDs.append(node.getID())
        potentialNewNodes = neuralNet2.getNodes()
        for node in potentialNewNodes:
            if not self.contains(insertedNodeIDs, node.getID()):
                newNodes.append(child.createNewNode([], -1, node.getID()))
                insertedNodeIDs.append(node.getID())

        #Determine the connections for the new network
        connections = neuralNet1.getConnections()
        for connection in connections:
            newConnections.append([connection.getInputNode().getID(), connection.getOutputNode().getID(), connection.getWeight(), connection.getInnovation(), connection.isEnabled()])
        connections = neuralNet2.getConnections()
        for connection in connections:
            innovation = connection.getInnovation()
            #If both neural nets have the same connection, average their weights
            foundConnection = False
            connectionIndex = 0
            while not foundConnection and connectionIndex < len(newConnections):
                if newConnections[connectionIndex][3] == innovation:
                    foundConnection = True
                else:
                    connectionIndex += 1
            if foundConnection:
                newConnections[connectionIndex][2] = (newConnections[connectionIndex][2] + connection.getWeight())/2
            #Otherwise, insert the new connection into the list
            else:
                newConnections.insert(innovation - 1, [connection.getInputNode().getID(), connection.getOutputNode().getID(), connection.getWeight(), connection.getInnovation(), connection.isEnabled()])

        #Actually make the connections
        for connection in newConnections:
            inNode = newNodes[self.findNode(newNodes, connection[0])]
            outNode = newNodes[self.findNode(newNodes, connection[1])]
            newConnection = Connection(connection[2], inNode, outNode, connection[3])
            isEnabled = connection[4]
            child.addConnection(newConnection)
            if not isEnabled:
                newConnection.disable()

        return child

    #A helper method to find a node by its nodeID in a list of nodes
    def findNode(self, array, nodeID):
        found = False
        arrayIndex = 0
        while (not found) and (arrayIndex < len(array)):
            arrayItem = array[arrayIndex]
            found = arrayItem.getID() == nodeID
            if not found:
                arrayIndex += 1
        return arrayIndex

    #A helper method to determine if an array contains an item
    def contains(self, array, item):
        contains = False
        arrayIndex = 0
        while (not contains) and (arrayIndex < len(array)):
            arrayItem = array[arrayIndex]
            contains = arrayItem == item
            arrayIndex += 1
        return contains

    #The method to mutate a neural net. The neural net can have three different types of mutations:
    #   -An existing connection's weight can be changed
    #   -A new node can be inserted
    #   -A new connection can be inserted
    #
    #Inputs: The neural net to be mutated
    #
    #Outputs: None
    def mutate(self, neuralNet):
        isOrdered, offNodeIn, offNodeOut = neuralNet.isOrdered()
        if not isOrdered:
            print("Ordering issue PRE-mutation:")
            print("In Node: " + str(offNodeIn))
            print("Out Node: " + str(offNodeOut))
            print("Nodes:")
            for node in neuralNet.getNodes():
                print(node.getID())
            print("---------------")
        #Potentially mutate an existing connection's weight
        self.existingConnectionMutation(self.existingConnectionMutationChance, neuralNet)
        isOrdered, offNodeIn, offNodeOut = neuralNet.isOrdered()
        if not isOrdered:
            print("Ordering issue Post-existingConnectionMutation:")
            print("In Node: " + str(offNodeIn))
            print("Out Node: " + str(offNodeOut))
            print("Nodes:")
            for node in neuralNet.getNodes():
                print(node.getID())
            print("---------------")
        #Potentially add a new node
        self.nodeMutation(self.nodeMutationChance, neuralNet)
        isOrdered, offNodeIn, offNodeOut = neuralNet.isOrdered()
        if not isOrdered:
            print("Ordering issue Post-nodeMutation:")
            print("In Node: " + str(offNodeIn))
            print("Out Node: " + str(offNodeOut))
            print("Nodes:")
            for node in neuralNet.getNodes():
                print(node.getID())
            print("---------------")
        #Potentially add a new connection
        self.connectionMutation(self.connectionMutationChance, neuralNet)
        isOrdered, offNodeIn, offNodeOut = neuralNet.isOrdered()
        if not isOrdered:
            print("Ordering issue Post-connectionMutation:")
            print("In Node: " + str(offNodeIn))
            print("Out Node: " + str(offNodeOut))
            print("Nodes:")
            for node in neuralNet.getNodes():
                print(node.getID())
            print("---------------")

    #The helper method to mutate an existing connection in a neural net. This method will change the weight of a random
    #connection in the neural net with the probability given by the mutationChance parameter
    #
    #Inputs: mutationChance - The chance that an existing connection is mutated (i.e. 0.8 is 80%)
    #        neuralNet - The neural net to potentially be mutated
    #
    #Outputs: None
    def existingConnectionMutation(self, mutationChance, neuralNet):
        connections = neuralNet.getConnections()
        for connection in connections:
            random.seed()
            chance = 0.01 * (1.0 * random.randrange(0, 100))
            if chance < mutationChance:
                connection.setWeight(0.1 * (1.0 * random.randrange(-10, 10)))
            
    #The helper method to mutate a neural net by potentially adding a new node. This mutation will occur
    #with the probability given by the mutationChance parameter
    #
    #Inputs: mutationChance - The chance that an existing connection is mutated (i.e. 0.8 is 80%)
    #        neuralNet - The neural net to potentially be mutated
    #
    #Outputs: None
    def nodeMutation(self, mutationChance, neuralNet):
        random.seed()
        chance = 0.01 * (1.0 * random.randrange(0, 100))
        if chance < mutationChance:
            connections = neuralNet.getEnabledConnections()
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
                if (connection.getInputNode().getID() == inNodeID) and (connection.getOutputNode().getID() == outNodeID):
                    newInnovation = False
                    nodeID = nodeInnovation[2]
                    innovation = nodeInnovation[3]
                nodeIndex += 1

            #Add a new node with the appropriate values
            neuralNet.insertNewNode(connection, innovation, nodeID)
            
            #If this innovation is new, record it and update self.innovationNumber and self.nodeID
            if newInnovation:
                inNodeID = connection.getInputNode().getID()
                outNodeID = connection.getOutputNode().getID()
                self.newNodesThisGeneration.append([inNodeID, outNodeID, self.nodeID, self.innovationNumber])
                self.innovationNumber += 2
                self.nodeID += 1

    #The helper method to mutate a neural net by potentially adding a new connection. This mutation will occur
    #with the probability given by the mutationChance parameter
    #
    #Inputs: mutationChance - The chance that an existing connection is mutated (i.e. 0.8 is 80%)
    #        neuralNet - The neural net to potentially be mutated
    #
    #Outputs: None
    def connectionMutation(self, mutationChance, neuralNet):
        random.seed()
        chance = 0.01 * (1.0 * random.randrange(0, 100))
        if chance < mutationChance:
            possibleConnections = neuralNet.getNonexistantConnections(self.innovationNumber)
            if len(possibleConnections) > 0:
                connection = possibleConnections[random.randrange(len(possibleConnections))]
                
                #Check if this innovation has occurred in this generation
                newInnovation = True
                connectionIndex = 0
                while newInnovation and connectionIndex < len(self.newConnectionsThisGeneration):
                    connectionInnovation = self.newConnectionsThisGeneration[connectionIndex]
                    inNodeID = connectionInnovation[0]
                    outNodeID = connectionInnovation[1]
                    if (connection.getInputNode().getID() == inNodeID) and (connection.getOutputNode().getID() == outNodeID):
                        newInnovation = False
                        connection.setInnovation(connectionInnovation[2])
                    connectionIndex += 1

                #Add the new connection
                neuralNet.addConnection(connection)
                
                #If this innovation is new, record it and update self.innovationNumber
                if newInnovation:
                    inNodeID = connection.getInputNode().getID()
                    outNodeID = connection.getOutputNode().getID()
                    self.newConnectionsThisGeneration.append([inNodeID, outNodeID, self.innovationNumber])
                    self.innovationNumber += 1

    #A helper method to perform rounding as it is traditionally done in school (i.e. x.5 rounds up to x+1 always)
    def round(self, num):
        return int(Context(rounding=ROUND_HALF_UP).to_integral_exact(Decimal(num)))

    def setNeuralNets(self, networks):
        self.networks = networks
