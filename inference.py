# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import util
import random
import busters
import game

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated Manhattan distance to the ghost you
        are tracking.

        The emissionModel below stores the probability of the noisyDistance for
        any true distance you supply. That is, it stores P(noisyDistance |
        TrueDistance).

        self.legalPositions is a list of the possible ghost positions (you
        should only consider positions that are in self.legalPositions).

        A correct implementation will handle the following special case:
          *  When a ghost is captured by Pacman, all beliefs should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).
        """
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"
        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
        allPossible = util.Counter()
        jail_position = self.getJailPosition()
        #  well the comment is hinting us to check if noisydistance is null and then
        # handle the jail case,  else we just update the observation multiplying the belief of a particular
        # position with emissionModel of the manhattan distance from pacman to that position.
        if noisyDistance is None:
            # pacman captured a ghost so place the ghost in the jail position
            # and now it must be in the jail position (the ghost) so make its probabilty 1.0
            allPossible[jail_position] = 1.0

        elif noisyDistance is not None:
            for p in self.legalPositions:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                if emissionModel[trueDistance] > 0:
                    # pacman belief is multiplied by the probability of ghost being at the position p
                    # then it is updated
                    new_belief = (self.beliefs[p] / 2) * emissionModel[trueDistance]
                    allPossible[p] = new_belief
                # we make allPossible equal to the observations that we make which is
                # the probabilty of pacman's belief that ghost is at a noisy distance multiplied with
                # the current belief and then below, we normalize it and make allPossible equal to self.beliefs

        "*** END YOUR CODE HERE ***"
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        """
        Update self.beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position (e.g., for DirectionalGhost).  However, this
        is not a problem, as Pacman's current position is known.

        In order to obtain the distribution over new positions for the ghost,
        given its previous position (oldPos) as well as Pacman's current
        position, use this line of code:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        Note that you may need to replace "oldPos" with the correct name of the
        variable that you have used to refer to the previous ghost position for
        which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in
        self.legalPositions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

        (and also given Pacman's current position).  You may also find it useful
        to loop over key, value pairs in newPosDist, like:

          for newPos, prob in newPosDist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern yourself),
        the line of code at the top of this comment block for obtaining
        newPosDist makes use of two helper methods provided in InferenceModule
        above:

          1) self.setGhostPosition(gameState, ghostPosition)
              This method alters the gameState by placing the ghost we're
              tracking in a particular position.  This altered gameState can be
              used to query what the ghost would do in this position.

          2) self.getPositionDistribution(gameState)
              This method uses the ghost agent to determine what positions the
              ghost will move to from the provided gameState.  The ghost must be
              placed in the gameState with a call to self.setGhostPosition
              above.

        It is worthwhile, however, to understand why these two helper methods
        are used and how they combine to give us a belief distribution over new
        positions after a time update from a particular position.
        """
        "*** YOUR CODE HERE ***"
        # start as same as the observe function. Call the util.Counter()
        allPossible = util.Counter()
        for p in self.legalPositions:
            #get the distribution of new positions as hinted in comments above:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, p))
            #again code hints for the next line too,
            #to loop over key, value pairs in newPosDist:
            for newPos, prob in newPosDist.items():
                new_belief = (self.beliefs[p] / 2) * prob
                # then update the new belief for each newPos, adding them up
                allPossible[newPos] = allPossible[newPos] + new_belief

        # normalize the allPossible and then set it to self.beliefs as we did in the observe function above
        allPossible.normalize()
        self.beliefs = allPossible

    def getBeliefDistribution(self):
        return self.beliefs




class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        """
        Initializes a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        "*** YOUR CODE HERE ***"
        self.particle_list = list()
        k = 0
        # A particle (sample) is a ghost position in this inference problem. and not a counter
        # so that is why we do not use util.Counter()
        while k < self.numParticles:
            for p in self.legalPositions:
                # for each particle we have, we will attach the particle to a position and
                # do this repeatedly until no more particles are left.
                self.particle_list.append(p)
                k += 1


    def observe(self, observation, gameState):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        "*** YOUR CODE HERE ***"

        # When a ghost is eaten, you should update all particles to place that ghost in its prison cell,
        #  as described in the comments of observe.
        if noisyDistance is None:
            i = 0
            while i < len(self.particle_list):
                self.particle_list[i] = self.getJailPosition()
                i += 1
        elif noisyDistance is not None:
            allPossible = util.Counter()

            current_belief = util.Counter()
            for particle in self.particle_list:
                current_belief[particle] += 1
            current_belief.normalize()

            for p in self.legalPositions:
                # You may also want to use util.manhattanDistance:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                allPossible[p] += emissionModel[trueDistance] * current_belief[p]

            # handling the case When all your particles receive zero weight based on the evidence,
            #  you should resample all particles from the prior to recover.

            if 0 == allPossible.totalCount(): # all particles are 0 so recover from the prior particles
                self.initializeUniformly(gameState)
            elif 0 != allPossible.totalCount():
                count = 0
                self.particle_list = list()
                while count < self.numParticles:
                    #util.sample(Counter object) is a helper method to generate a sample from a belief distribution.
                    #  creating sample from allPossible counter object
                    self.particle_list.append(util.sample(allPossible))
                    count += 1


    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        "*** YOUR CODE HERE ***"
        k = 0
        while k < len(self.particle_list):
            # as again hinted above we use newposdist:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, self.particle_list[k]))
            #util.sample(Counter object) is a helper method to generate a sample from a belief distribution:
            # time elapses so we will resample according to our new position distribution counter
            self.particle_list[k] = util.sample(newPosDist)
            k += 1


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        "*** YOUR CODE HERE ***"
        # counter object to be returned
        allPossible = util.Counter()
        # then for each position in our particle list increment the counter of that key's value
        #  by 1 and then normalize and return the counter object
        for p in self.particle_list:
            if p:
                allPossible[p] += 1
            elif not p:
                continue
        allPossible.normalize()
        return allPossible

class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1:
            jointInference.observeState(gameState)

    def elapseTime(self, gameState):
        "Update beliefs for a time step elapsing from a gameState."
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.numParticles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the `itertools.product` function to get an
        implementation of the Cartesian product.

        Note: If you use itertools, keep in mind that permutations are not
        returned in a random order; you must shuffle the list of permutations in
        order to ensure even placement of particles across the board. Use
        self.legalPositions to obtain a list of positions a ghost may occupy.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        "*** YOUR CODE HERE ***"

        # I followed the hint and used the `itertools.product` function to get an
        # implementation of the Cartesian product

        # Repeat is used so that if we have more than 2 ghosts it will work
        # but in test cases we always have 2 ghosts.
        number_of_ghosts = self.numGhosts
        legal_ghost_positions = itertools.product(self.legalPositions, repeat=number_of_ghosts)

        # The above line returned a product object so we convert it to list with the following line:
        legal_ghost_positions = list(legal_ghost_positions)

        # permutations are not returned in a random order so we shuffle the list.
        random.shuffle(legal_ghost_positions)

        self.particle_list = list()
        k = 0
        # A particle (sample) is a ghost position in this inference problem. It is not a counter
        # so that is why we do not use util.Counter() for particle_list
        while k < self.numParticles:
            if k < len(legal_ghost_positions):
                self.particle_list.append(legal_ghost_positions[k])
            elif k == len(legal_ghost_positions):
                self.particle_list.append(legal_ghost_positions[0])
            elif k > len(legal_ghost_positions):
                i = k % len(legal_ghost_positions)
                self.particle_list.append(legal_ghost_positions[i])
            k += 1
        return self.particle_list

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observeState(self, gameState):
        """
        Resamples the set of particles using the likelihood of the noisy
        observations.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition(i) where `i` is the index of the ghost.

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeParticles. After all
             particles are generated randomly, any ghosts that are eaten (have
             noisyDistance of None) must be changed to the jail Position. This
             will involve changing each particle if a ghost has been eaten.

        self.getParticleWithGhostInJail is a helper method to edit a specific
        particle. Since we store particles as tuples, they must be converted to
        a list, edited, and then converted back to a tuple. This is a common
        operation when placing a ghost in jail.
        """
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()
        if len(noisyDistances) < self.numGhosts:
            return
        emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

        "*** YOUR CODE HERE ***"
        allPossible = util.Counter()

        # self.particle_list is how we handle the problem of having more particles than
        # the ghost positions

        for p in self.particle_list:
            k = 0
            while k < self.numGhosts:
                if k == 0:
                    weight = int(1)
                # if we ate the ghost:
                if noisyDistances[k] is None:
                    # placing a ghost in jail :
                    p = self.getParticleWithGhostInJail(p, k)
                # if we have not eaten the ghost then we update the probabilities after the observation
                elif noisyDistances[k] is not None:
                    # distance from ghost to pacman
                    trueDistance = util.manhattanDistance(p[k], pacmanPosition)
                    # To get a particle weight, you need to multiply the emission probabilities together
                    # the next line is hinted in the PDF file of the project
                    if k == 0:
                        weight = 1 * emissionModels[0][trueDistance]
                    elif k != 0:
                        weight = weight * emissionModels[k][trueDistance]
                k += 1
            allPossible[p] = allPossible[p] + weight


        # FOLLOWING PART OF THE CODE IS THE SAME AS THE ParticleFilter CODE'S SECOND PART, NOTHING IS CHANGED:

        # handling the case When all your particles receive zero weight based on the evidence,
        #  you should resample all particles from the prior to recover.

        if 0 == allPossible.totalCount():  # all particles are 0 so recover from the prior particles
            self.initializeParticles()
        elif 0 != allPossible.totalCount():
            count = 0
            while count < self.numParticles:
                # util.sample(Counter object) is a helper method to generate a sample from a belief distribution.
                #  creating sample from allPossible counter object
                self.particle_list[count] = util.sample(allPossible)
                count += 1

    def getParticleWithGhostInJail(self, particle, ghostIndex):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghostIndex'th ghost in jail.
        """
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        """
        Samples each particle's next state based on its current state and the
        gameState.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        Then, assuming that `i` refers to the index of the ghost, to obtain the
        distributions over new positions for that single ghost, given the list
        (prevGhostPositions) of previous positions of ALL of the ghosts, use
        this line of code:

          newPosDist = getPositionDistributionForGhost(
             setGhostPositions(gameState, prevGhostPositions), i, self.ghostAgents[i]
          )

        Note that you may need to replace `prevGhostPositions` with the correct
        name of the variable that you have used to refer to the list of the
        previous positions of all of the ghosts, and you may need to replace `i`
        with the variable you have used to refer to the index of the ghost for
        which you are computing the new position distribution.

        As an implementation detail (with which you need not concern yourself),
        the line of code above for obtaining newPosDist makes use of two helper
        functions defined below in this file:

          1) setGhostPositions(gameState, ghostPositions)
              This method alters the gameState by placing the ghosts in the
              supplied positions.

          2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
              This method uses the supplied ghost agent to determine what
              positions a ghost (ghostIndex) controlled by a particular agent
              (ghostAgent) will move to in the supplied gameState.  All ghosts
              must first be placed in the gameState using setGhostPositions
              above.

              The ghost agent you are meant to supply is
              self.ghostAgents[ghostIndex-1], but in this project all ghost
              agents are always the same.
        """
        newParticles = []
        for oldParticle in self.particle_list:
            newParticle = list(oldParticle) # A list of ghost positions
            # now loop through and update each entry in newParticle...

            "*** YOUR CODE HERE ***"

            # this is almost identical to elapsetime function of ParticleFilter with only exception
            # being that instead of one ghost we have two ghosts:

            k = 0
            while k < self.numGhosts:

                # I will again use the hinted function getPositionDistributionforGhost which gives the
                # distribution over new positions of ghosts
                newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, newParticle), k, self.ghostAgents[k])

                # then go to the kth element of newParticle list and update it via sampling
                # in other words; updating each entry in newParticle

                newParticle[k] = util.sample(newPosDist)
                k += 1
            "*** END YOUR CODE HERE ***"

            newParticles.append(tuple(newParticle))
        self.particle_list = newParticles

    def getBeliefDistribution(self):
        "*** YOUR CODE HERE ***"

        # THIS IS AGAIN THE SAME AS ParticleFilter's getBeliefDistribution function nothing changes:

        allPossible = util.Counter()
        # then for each position in our particle list increment the counter of that key's value
        #  by 1 and then normalize and return the counter object
        for p in self.particle_list:
            if p:
                allPossible[p] += 1
            elif not p:
                continue
        allPossible.normalize()
        return allPossible


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState
