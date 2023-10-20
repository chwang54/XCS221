from typing import Callable, List, Set

import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model


class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        pass
        # ### START CODE HERE ###
        return 0
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        return state == len(self.query)
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        successors = []
        for end in range(state+1,len(self.query)+1):
            next_word = self.query[state:end]
            cost = self.unigramCost(next_word)
            successors.append((next_word,end,cost))
        return successors
        # ### END CODE HERE ###


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ""

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # ### START CODE HERE ###
    if ucs.totalCost is None:
        return ""
    
    return ' '.join(ucs.actions)
    # ### END CODE HERE ###


############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost


class VowelInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        queryWords: List[str],
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        return (0, "-BEGIN-")
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        return state[0] == len(self.queryWords)
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        word_index, last_filled_word = state
        successors = []

        next_word = self.queryWords[word_index]
        fills = self.possibleFills(next_word)
        for fill in fills:
            next_state = (word_index+1,fill)
            cost = self.bigramCost(last_filled_word,fill)
            successors.append((fill,next_state,cost))
        return successors
        # ### END CODE HERE ###


def insertVowels(
    queryWords: List[str],
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    pass
    # ### START CODE HERE ###
    problem = VowelInsertionProblem(queryWords,bigramCost,possibleFills)
    ucs=util.UniformCostSearch()
    ucs.solve(problem)

    return ' '.join(ucs.actions) if ucs.actions else None
    # ### END CODE HERE ###


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem


class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        query: str,
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###


def segmentAndInsert(
    query: str,
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    if len(query) == 0:
        return ""

    # ### START CODE HERE ###
    # ### END CODE HERE ###


############################################################

if __name__ == "__main__":
    shell.main()
