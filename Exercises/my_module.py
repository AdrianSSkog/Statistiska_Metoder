import numpy as np


def congruent(a, b, p):
    if p == 0:
        raise ValueError("p can not be 0")
    return a % p == b

def divisor(a, b):
    if a == 0:
        raise ValueError("a can not be 0")
    return b % a == 0

def prime(a):

    if a > 1:
        for i in range(2, a):
            if a % i == 0:
                return False
        return True
    else:
        return False

def test_congruence():
    a = congruent(5,2,3)
    b = not congruent(1,4,2)
    return a and b

def test_divisor():
    a = divisor(2,12)
    b = not divisor(5,7)
    return a and b

def test_prime():
    a = prime(37)
    b = not prime(8)
    return a and b

class DataHandler:
    def __init__(self, path : str):
        self.path = path
        self._data = None
        self._header = None
        self._col_index = None

    def _read_file(self):
        if self._data is None:
            with open(self.path, "r") as file:
                self._data = [r.split(",") for r in file.read().splitlines()]
                self._header = [r.strip('"') for r in self._data[0]]
                self._col_index = dict(zip(self._header, list(range(len(self._header)))))
                self._data = np.array([list(map(lambda x: float(x.strip('"')), row)) for row in self._data[1:]])
        return self._data
    
    @classmethod
    def read_file(cls, path : str):
        obj = cls(path)
        obj._read_file()
        return obj
    
    def get_headers(self):
        self._read_file()
        return self._header

    def get_head(self, rows=5):
        self._read_file()
        return self._header, self._data[:rows]        
    
    def get_row(self, index):
        return self._read_file()[index]
    
    def get_column(self, index):
        data = self._read_file()
        if isinstance(index, str):
            if index in self._header:
                index = self._col_index[index]
            else:
                raise KeyError(f"There is no column with the name {index}")
        return data[:,index]
    
    def pretty_print(self):
        self._read_file()
        for name in self._header:
            if not name:
                continue
            col = self.get_column(name)
            print(f"{name}")
            print(col)
            print()

if __name__=="__main__":
    MyObject = DataHandler("Advertising.csv")
    MyObject.pretty_print()


