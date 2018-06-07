from time import process_time

flag = 0

def begin():
    flag = process_time()

def end():
    return process_time() - flag