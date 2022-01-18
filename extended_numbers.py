#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numbers
# Complex
class C:

    def __init__(self, data):
        self.data = np.array(data)

    def __repr__(self):
        return "%s + %s i" %(self.data[0], self.data[1])

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __add__(self, other):
        return C( self.data + other.data )

    def __neg__(self):
        return C( -self.data )

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, C):
            return C( (self.data[0]*other.data[0] - self.data[1]*other.data[1],
                       self.data[0]*other.data[1] + self.data[1]*other.data[0]) )
        elif isinstance(other, numbers.Number):
            return C( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return C( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")

# Quaternion
class Q:

    def __init__(self, data):
        self.data = np.array(data)

    def __repr__(self):
        return "%s + %s i + %s j + %s k" %(self.data[0], self.data[1], self.data[2], self.data[3])

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __add__(self, other):
        return Q( self.data + other.data )

    def __neg__(self):
        return Q( -self.data )

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, Q):
            return Q( (self.data[0]*other.data[0] - self.data[1]*other.data[1] - self.data[2]*other.data[2] - self.data[3]*other.data[3],
                       self.data[0]*other.data[1] + self.data[1]*other.data[0] + self.data[2]*other.data[3] - self.data[3]*other.data[2],
                       self.data[0]*other.data[2] - self.data[1]*other.data[3] + self.data[2]*other.data[0] + self.data[3]*other.data[1],
                       self.data[0]*other.data[3] + self.data[1]*other.data[2] - self.data[2]*other.data[1] + self.data[3]*other.data[0]) )
        elif isinstance(other, numbers.Number):
            return Q( other*self.data )
        else:
            raise ValueError("Can only multiply quaternion numbers by other quaternion numbers or by scalars")

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Q( other*self.data )
        else:
            raise ValueError("Can only multiply quaternion numbers by other quaternion numbers or by scalars")

# Octonion
class O:

    def __init__(self, data):
        self.data = np.array(data)

    def __repr__(self):
        return "%s e0 + %s e1 + %s e2 + %s e3 + %s e4 + %s e5 + %s e6 + %s e7" %(self.data[0], self.data[1], self.data[2], self.data[3], self.data[4], self.data[5], self.data[6], self.data[7])

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __add__(self, other):
        return O( self.data + other.data )

    def __neg__(self):
        return O( -self.data )

    def __sub__(self, other):
        return self + (-other)

    def mul(other):
        if isinstance(other, O):
            return O( (self.data[0]*other.data[0] - self.data[1]*other.data[1] - self.data[2]*other.data[2] - self.data[3]*other.data[3] - self.data[4]*other.data[4] - self.data[5]*other.data[5] - self.data[6]*other.data[6] - self.data[7]*other.data[7],
                       self.data[0]*other.data[1] + self.data[1]*other.data[0] + self.data[2]*other.data[3] - self.data[3]*other.data[2] + self.data[4]*other.data[5] - self.data[5]*other.data[4] - self.data[6]*other.data[7] + self.data[7]*other.data[6],
                       self.data[0]*other.data[2] - self.data[1]*other.data[3] + self.data[2]*other.data[0] + self.data[3]*other.data[1] + self.data[4]*other.data[6] + self.data[5]*other.data[7] - self.data[6]*other.data[4] - self.data[7]*other.data[5],
                       self.data[0]*other.data[3] + self.data[1]*other.data[2] - self.data[2]*other.data[1] + self.data[3]*other.data[0] + self.data[4]*other.data[7] - self.data[5]*other.data[6] + self.data[6]*other.data[5] - self.data[7]*other.data[4],
                       self.data[0]*other.data[4] - self.data[1]*other.data[5] - self.data[2]*other.data[6] - self.data[3]*other.data[7] + self.data[4]*other.data[0] + self.data[5]*other.data[1] + self.data[6]*other.data[2] + self.data[7]*other.data[3],
                       self.data[0]*other.data[5] + self.data[1]*other.data[4] - self.data[2]*other.data[7] + self.data[3]*other.data[6] - self.data[4]*other.data[1] + self.data[5]*other.data[0] - self.data[6]*other.data[3] + self.data[7]*other.data[2],
                       self.data[0]*other.data[6] + self.data[1]*other.data[7] + self.data[2]*other.data[4] - self.data[3]*other.data[5] - self.data[4]*other.data[2] + self.data[5]*other.data[3] + self.data[6]*other.data[0] - self.data[7]*other.data[1],
                       self.data[0]*other.data[7] - self.data[1]*other.data[6] + self.data[2]*other.data[5] + self.data[3]*other.data[4] - self.data[4]*other.data[3] - self.data[5]*other.data[2] + self.data[6]*other.data[1] + self.data[7]*other.data[0]) )
        else:
            raise ValueError("Can only multiply octonion numbers by other octonion numbers in mul method object")
    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return O( other*self.data )
        else:
            raise ValueError("Can only multiply octonion numbers by scalars in * operator")

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return O( other*self.data )
        else:
            raise ValueError("Can only multiply octonion numbers by scalars in * operator")
