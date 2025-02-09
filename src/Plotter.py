from sklearn import tree
from matplotlib import pyplot as plt

labels = X_train.columns

figure = plt.figure(figsize=(25,25))
dinges = tree.plot_tree(model,
                        feature_names= labels,
                        class_names={0: 'passed_away', 1:'survived'},
                        filled = True,
                        fontsize = 10)
