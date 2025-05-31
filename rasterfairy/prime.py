#
# Prime Number Utility v1.02
# part of Raster Fairy v1.0,
# released 22.01.2016
#
# The purpose of Raster Fairy is to transform any kind of 2D point cloud into
# a regular raster whilst trying to preserve the neighborhood relations that
# were present in the original cloud. If you feel the name is a bit silly and
# you can also call it "RF-Transform".
#
# NOTICE: if you use this algorithm in an academic publication, paper or 
# research project please cite it either as "Raster Fairy by Mario Klingemann" 
# or "RF-Transform by Mario Klingemann"
#
#
# 
# Copyright (c) 2016, Mario Klingemann, mario@quasimondo.com
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Mario Klingemann nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MARIO KLINGEMANN BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module provides utility classes and functions for working with prime numbers
and permutations.
"""

import math
from collections import Counter
from itertools import permutations

class PrimeNumber:
    """Represents a node in a linked list of prime numbers."""
    def __init__(self, n):
        self.n = n
        self.nextPrime = None
    
    def setNext(self, n):
        self.nextPrime = PrimeNumber(n)
        return self.nextPrime

class Prime:
    """Provides methods for prime number generation, factorization, and permutations."""
    
    def __init__(self, max_permutations=1000000):
        self.firstPrime = PrimeNumber(2)
        self.lastPrime = self.firstPrime.setNext(3).setNext(5).setNext(7)
        self.max_permutations = max_permutations

    def getPrimeFactors(self, n):
        """Calculates the prime factors of a given integer."""
        if n <= 1:
            return []
        
        result = []
        factor = 2
        while n > 1:
            if n % factor == 0:
                result.append(factor)
                n //= factor
            else:
                factor = self.nextPrime(factor)
        return result

    def isPrime(self, n):
        """Checks if a given integer is a prime number."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:  # Even numbers > 2
            return False
        if n == 3:
            return True
        if n % 3 == 0:
            return False
        if n == 5:
            return True
        if n % 5 == 0:  # Multiples of 5 > 5
            return False

        maxCheck = math.sqrt(n)
        if maxCheck == math.floor(maxCheck):
            return False
        
        p = self.firstPrime
        while p is not None:
            if p.n > maxCheck:
                return True
            if n % p.n == 0:
                return False
            p = p.nextPrime

        divisor = self.lastPrime.n + 2
        while divisor <= maxCheck:
            if not self._isPrimeSimple(divisor):  # Avoid recursion
                divisor += 2
                continue
            self.lastPrime = self.lastPrime.setNext(divisor)
            if divisor > maxCheck:
                return True
            if n % divisor == 0:
                return False
            divisor += 2
        return True
    
    def _isPrimeSimple(self, n):
        """Simple primality test without extending the prime list."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        sqrt_n = math.sqrt(n)
        p = self.firstPrime
        while p is not None and p.n <= sqrt_n:
            if n % p.n == 0:
                return False
            p = p.nextPrime
        
        # Check remaining odd numbers up to sqrt(n)
        divisor = self.lastPrime.n + 2 if self.lastPrime.n < sqrt_n else int(sqrt_n) + 1
        while divisor <= sqrt_n:
            if n % divisor == 0:
                return False
            divisor += 2
        return True
    
    def nextPrime(self, n):
        """Finds the next prime number greater than n."""
        if n < 2:
            return 2
        
        # Start checking from n+1, or n+2 if n is even
        candidate = n + 1
        if candidate % 2 == 0:
            candidate += 1
        
        # Check if we already have this prime in our list
        if self.lastPrime.n >= candidate:
            p = self.firstPrime
            while p is not None:
                if p.n > n:
                    return p.n
                p = p.nextPrime
        
        # Find the next prime
        while not self.isPrime(candidate):
            candidate += 2
        return candidate
    
    def getPermutations(self, symbols):
        """Generates all unique permutations with memory-conscious fallback."""
        if len(symbols) == 0:
            return [()]
        
        # Calculate actual number of unique permutations
        symbol_counts = Counter(symbols)
        n = len(symbols)
        
        # Calculate n! / (n1! * n2! * ... * nk!) for unique permutations
        unique_count = self.factorial(n)
        for count in symbol_counts.values():
            unique_count //= self.factorial(count)
        
        # If manageable, generate all permutations
        if unique_count <= self.max_permutations:
            return list(set(permutations(symbols)))
        
        # Fallback: Group symbols and generate representative permutations
        print(f"Not enough memory for {unique_count:,} permutations, creating grouped set")
        return self._getGroupedPermutations(symbol_counts)
    
    def _getGroupedPermutations(self, symbol_counts):
        """Generate permutations by treating repeated symbols as groups."""
        groups = list(symbol_counts.items())
        
        if len(groups) == 1:
            # Only one type of symbol
            symbol, count = groups[0]
            return [tuple([symbol] * count)]
        
        # Generate permutations of the groups
        group_perms = list(set(permutations(groups)))
        
        # Convert back to symbol lists
        result = []
        for group_perm in group_perms:
            expanded = []
            for symbol, count in group_perm:
                expanded.extend([symbol] * count)
            result.append(tuple(expanded))
        
        return result
    
    def factorial(self, n):
        """Calculates the factorial of a non-negative integer."""
        if n < 0:
            return 0
        r = 1
        while n > 1:
            r *= n
            n -= 1
        return r
    
    def getNthPermutation(self, symbols, n):
        """Generates the nth lexicographical permutation of a list of symbols."""
        return self.permutation(symbols, self.n_to_factoradic(n))

    def n_to_factoradic(self, n, p=2):
        """Converts a non-negative integer to its factoradic representation."""
        if n < p:
            return [n]
        ret = self.n_to_factoradic(n // p, p + 1)
        ret.append(n % p)
        return ret
    
    def permutation(self, symbols, factoradic):
        """Generates a permutation of symbols based on its factoradic representation."""
        factoradic = factoradic[:]  # Make a copy to avoid modifying original
        factoradic.append(0)
        while len(factoradic) < len(symbols):
            factoradic = [0] + factoradic
        
        ret = []
        s = symbols[:]
        while len(factoradic) > 0:
            f = factoradic.pop(0)
            ret.append(s.pop(f))
        
        return tuple(ret)

