import util, submission

# Run this for both smallMDP and largeMDP
print("Comparing Q-learning and Value Iteration for smallMDP:")
different_actions_small = submission.simulate_QL_over_MDP(submission.smallMDP, submission.identityFeatureExtractor)
# print("Comparing Q-learning and Value Iteration for largeMDP:")
# different_actions_large = submission.simulate_QL_over_MDP(submission.largeMDP, submission.identityFeatureExtractor)