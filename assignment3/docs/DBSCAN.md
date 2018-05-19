# DBSCAN

## Algorithms

Create a decision tree by repeating grouping by finding the rule(decision) that minimizes the given metric(default gini). Pretty simple and easy.

Support **random forest**. This is an ensemble method in which a plurality of decision trees are randomly learned. Divide learning data randomly, learn each tree, make several weak classifiers, and give the final result through voting. *max_features* can limit the number of features that each tree will use when creating random forest.

Support various **metrics** such as gini, entropy, and classification error.

Various options to **prevent overfitting** are supported such as *depth*, *min_size*, *min_gain* and *max_features*. You can limit the *depth* of the tree to avoid overfitting the tree.  Prevent the creation of small leaf nodes by entering the *min_size* at which the decision will be created. Also, by setting *min_gain*, you can avoid branching if you do not exceed the minimum information gain. This helps each tree to learn a different characteristic so it can get a better value.

## Implementations

Python is a simple language that is easy enough to understand directly. So it's not difficult to see and understand the code right away. But here are tips for Python beginners.

### `DBSCAN`



## Requirements

- NumPy: is the fundamental package for scientific computing with Python.
- Pandas: is providing high-performance, easy-to-use data structures and data analysis tools for the Python.

*install packages using pip*
```
pip3 install -r requirements.txt
```

*Tested @ python3.5 in Ubuntu 16.04 LTS, macOS High Sierra and Windows 10*

Run as below
```
python3 dt.py (input) (n) (eps) (min) [--output output_path] [--image]
```

## Performance

![input1](C:\Users\maybe\Documents\Workspace\ITE4005\assignment3\data\input1.png)

*input1*

![input2](C:\Users\maybe\Documents\Workspace\ITE4005\assignment3\data\input2.png)

*input2*

![input3](C:\Users\maybe\Documents\Workspace\ITE4005\assignment3\data\input3.png)

*input3*