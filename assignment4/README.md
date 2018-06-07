Recommender

Documents in /doc for assignment report.



## Result

|      | u1    | u2    | u3    | u4    | u5    |
| ---- | ----- | ----- | ----- | ----- | ----- |
| RMSE | 0.867 | 0.860 | 0.857 | 0.857 | 0.861 |

## Requirements

- NumPy: is the fundamental package for scientific computing with Python.
- Pandas: is providing high-performance, easy-to-use data structures and data analysis tools for the Python.
- Cython: support compiled language, generates CPython extension modules.

*install packages using pip*
```
pip3 install -r requirements.txt
```

*Tested @ python3.5 in Ubuntu 16.04 LTS, macOS High Sierra and Windows 10*

Run as below
```
// First of all, build Cython extensions
python setup.py build_ext --inplace

// Run recommender
python recommender.py [train_data_path] [test_data_path]
```
