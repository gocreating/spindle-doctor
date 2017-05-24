# http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def row_count(filename):
    return sum(1 for line in open(filename))
