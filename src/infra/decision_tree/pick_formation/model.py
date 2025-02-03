from sklearn import tree
import pickle

# defensor | defensor - midfielder | midfielder | attacker midfielder | foward | formation type

labels = [{"GK": 1, "D": 4, "DM": 0, "MC": 2, "AM": 0, "F": 4, "formation": "4-2-4"},
          {"GK": 1, "D": 3, "DM": 0, "M": 2, "AM": 2, "F": 3, "formation": "3-4-3"},
          {"GK": 1, "D": 4, "DM": 1, "MC": 2, "AM": 0, "F": 3, "formation": "4-3-3"},
          {"GK": 1, "D": 3, "DM": 1, "MC": 2, "AM": 2, "F": 2, "formation": "3-5-2"},
          {"GK": 1, "D": 4, "DM": 2, "MC": 0, "AM": 2, "F": 2, "formation": "4-4-2"},
          {"GK": 1, "D": 4, "DM": 1, "MC": 0, "AM": 3,
              "F": 2, "formation": "4-1-3-2"},
          {"GK": 1, "D": 4, "DM": 2, "MC": 0, "AM": 3,
              "F": 1, "formation": "4-2-3-1"},
          {"GK": 1, "D": 3, "DM": 2, "MC": 2, "AM": 1,
              "F": 2, "formation": "3-4-1-2"},
          {"GK": 1, "D": 5, "DM": 0, "MC": 3, "AM": 0, "F": 2, "formation": "5-3-2"},
          {"GK": 1, "D": 5, "DM": 1, "MC": 2, "AM": 1, "F": 1, "formation": "5-4-1"},
          {"GK": 1, "D": 4, "DM": 2, "MC": 3, "AM": 0, "F": 1, "formation": "4-5-1"},
          {"GK": 1, "D": 3, "DM": 2, "MC": 3, "AM": 1, "F": 1, "formation": "3-6-1"}]

inputs = [[8, 0, 4, 0, 8, 2], [6, 0, 4, 4, 6, 2],
          [8, 2, 4, 0, 6, 2], [6, 2, 4, 4, 4, 2],

          [8, 4, 0, 4, 4, 1], [8, 2, 0, 6, 4, 1],
          [8, 4, 0, 6, 2, 1], [6, 4, 4, 2, 4, 1],

          [10, 0, 6, 0, 4, 0], [10, 2, 4, 2, 2, 0],
          [8, 4, 6, 0, 2, 0], [6, 4, 6, 2, 2, 0]]

outputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(inputs, outputs)

with open('pick_formation.mdl', 'wb') as file:
    pickle.dump(clf, file)
