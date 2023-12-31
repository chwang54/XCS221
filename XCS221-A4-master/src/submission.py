from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    pass
    # ### START CODE HERE ###
    def minimax(agentIndex, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)
      if agentIndex == 0:  # Pac-Man, Maximizer
        return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
      else:  # Ghosts, Minimizer
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
          nextAgent = 0
          depth += 1
        return min(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))

    # Start of minimax
    bestScore = float("-inf")
    bestAction = Directions.STOP
    for action in gameState.getLegalActions(0):
      # Calculate the score of each action by considering the next agent and increasing the depth.
      score = minimax(1, 0, gameState.generateSuccessor(0, action))
      if score > bestScore:
        bestScore = score
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    pass
    # ### START CODE HERE ###
    def alphaBeta(agentIndex, depth, gameState, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState), None
        
      if agentIndex == 0:  # Pac-Man, Maximizer
        return maxValue(agentIndex, depth, gameState,alpha, beta)
      else:  # Ghosts, Minimizer
        return minValue(agentIndex, depth, gameState, alpha, beta)
       
    def maxValue(agentIndex, depth, gameState, alpha, beta):
      v = float("-inf")
      bestAction = None
      for action in gameState.getLegalActions(agentIndex):
        successorState = gameState.generateSuccessor(agentIndex, action)
        successorValue = alphaBeta((agentIndex + 1) % gameState.getNumAgents(), depth + 1, successorState, alpha, beta)[0]
        if successorValue > v:
          v,bestAction = successorValue, action
        if v > beta:
          return v,bestAction
        alpha = max(alpha,v)
      return v,bestAction

    def minValue(agentIndex, depth, gameState, alpha, beta):
      v = float("inf")
      bestAction = None
      for action in gameState.getLegalActions(agentIndex):
        successorState = gameState.generateSuccessor(agentIndex, action)
        successorValue = alphaBeta((agentIndex + 1) % gameState.getNumAgents(), depth + 1, successorState, alpha, beta)[0]
        if successorValue < v:
          v,bestAction = successorValue, action
        if v < alpha:
          return v,bestAction
        beta = min(beta,v)
      return v,bestAction
  
    # Initial call to alpha-beta
    _, action = alphaBeta(0, 0, gameState, float("-inf"), float("inf"))
    return action
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    pass
    # ### START CODE HERE ###
    def expectimax(agentIndex, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)
      if agentIndex == 0:  # Pac-Man, Maximizer
        return max(expectimax(1, depth + 1, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
      else:  # Ghosts, Expected Value
        nextAgent = agentIndex + 1
        if nextAgent >= gameState.getNumAgents():
          nextAgent = 0
          depth += 1
        actions = gameState.getLegalActions(agentIndex)
        return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in actions) / len(actions)

    # Start of expectimax
    bestScore = float("-inf")
    bestAction = Directions.STOP
    for action in gameState.getLegalActions(0):
      score = expectimax(1, 1, gameState.generateSuccessor(0, action))
      if score > bestScore:
        bestScore = score
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
    This evaluation function takes into account several features:
    - The score of the game state
    - The distance to the nearest food dot
    - The distance to the nearest (non-scared) ghost
    - The number of capsules left
    - The number of food dots left
    - The state of the ghosts (scared or not) and their proximity
  """
  pass
  ### START CODE HERE ###
  # Useful information you can extract from a GameState (pacman.py)
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  score = currentGameState.getScore()

  # Compute distance to the nearest food
  foodList = newFood.asList()
  minFoodDistance = min([util.manhattanDistance(newPos, food) for food in foodList]) if foodList else 1

  # Compute distance to ghosts and handle division by zero
  ghostDistances = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
  # Avoid division by zero by adding a small amount to the distance
  minGhostDistance = min(ghostDistances) if ghostDistances else 10
  ghostScore = sum((2 + scaredTime) / (distance + 0.1) for scaredTime, distance in zip(newScaredTimes, ghostDistances))

  # Number of capsules left
  numCapsulesLeft = len(currentGameState.getCapsules())

  # Encourage eating food by giving a positive score for being close to food
  foodScore = 1 / float(minFoodDistance) if minFoodDistance > 0 else 100

  # Encourage eating capsules by giving a higher score if fewer capsules are left
  capsuleScore = -3 * numCapsulesLeft

  # Encourage eating scared ghosts
  scaredGhostScore = ghostScore if minGhostDistance > 1 else -200

  # Features and weights
  features = [score, foodScore, scaredGhostScore, capsuleScore]
  weights = [1, 5, 2, 10]

  # Combine features with weights
  return sum(feature * weight for feature, weight in zip(features, weights))
### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction
