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

class PrimeNumber():
    """Represents a node in a linked list of prime numbers."""
    def __init__(self,n):
        """Initializes a PrimeNumber object.

        Args:
            n: An integer representing the prime number.
        """
        self.n = n
        self.nextPrime = None
    
    def setNext(self,n):
        """Sets the next prime number in the linked list.

        Args:
            n: An integer representing the next prime number.

        Returns:
            A new PrimeNumber object representing the next prime.
        """
        self.nextPrime = PrimeNumber(n)
        return self.nextPrime
    
class Prime():
    """Provides methods for prime number generation, factorization, and permutations."""
    firstPrime = PrimeNumber(2)
    lastPrime = firstPrime.setNext(3).setNext(5).setNext(7)

    def getPrimeFactors(self,n):
        """Calculates the prime factors of a given integer.

        Args:
            n: An integer for which to find prime factors.

        Returns:
            A list of integers representing the prime factors of n.
        """
        result = []
        factor = 2
        while n > 1:
            if n % factor == 0:
                result.append(factor)
                n //= factor
            else:
                factor = self.nextPrime(factor)
                
        return result

    def isPrime(self,n):
        """Checks if a given integer is a prime number.

        This method uses a combination of trial division by small primes
        and then checks divisibility by subsequent odd numbers up to the
        square root of n. It also maintains a linked list of primes found
        so far to speed up checks for larger numbers.

        Args:
            n: An integer to check for primality.

        Returns:
            True if n is a prime number, False otherwise.
        """
        if (n & 1) == 0 or (n > 5 and n % 5 == 0):
            return False

        maxCheck = math.sqrt(n)
        if maxCheck == math.floor(maxCheck):
            return False
        
        p = self.firstPrime
        while p != None:
            if p.n > maxCheck:
                return True
            if n % p.n == 0:
                return False
            p = p.nextPrime

        divisor = self.lastPrime.n + 2
        while divisor <= maxCheck:
            if not self.isPrime(divisor):
                divisor += 2
                continue
            self.lastPrime = self.lastPrime.setNext(divisor)
            if divisor > maxCheck:
                return True
            if n % divisor == 0:
                return False
            divisor += 2
        return True
    
    def nextPrime(self,n):
        """Finds the next prime number greater than or equal to n.

        Args:
            n: An integer from which to start searching for the next prime.

        Returns:
            An integer representing the next prime number.
        """
        n += (n&1)+1
        if self.lastPrime.n > n:
            p = self.firstPrime
            while True:
                if p.n >= n:
                    return p.n
                p = p.nextPrime
        else:
            while not self.isPrime(n):
                n += 2
        return n
    
    def getPermutations(self,symbols):
        """Generates all unique permutations of a list of symbols.

        If the number of symbols is less than 10, it generates all
        permutations directly. Otherwise, if there are repeated symbols,
        it groups them and generates permutations of the groups to
        avoid memory issues.

        Args:
            symbols: A list of items (symbols) to permute.

        Returns:
            A list of tuples, where each tuple is a unique permutation
            of the input symbols.
        """
        if len(symbols) < 10:
            n = self.factorial(len(symbols))
            if n==1:
                return [tuple(symbols[:])]

            perm = []
            for i in range(n):
                perm.append(self.getNthPermutation(symbols, i))
               
        else:
            print("not enough memory for amount of possible permutations, creating grouped set")
            groupedSymbols = []
            lastSymbol = symbols[0]
            c = 1
            for i in range(1,len(symbols)):
                if symbols[i]==lastSymbol:
                    c+=1
                else:
                    groupedSymbols.append((lastSymbol,c))
                    c= 1
                    lastSymbol = symbols[i]
            groupedSymbols.append((lastSymbol,c))
            n = self.factorial(len(groupedSymbols))
            if n==1:
                return [tuple(symbols[:])]

            perm = []
            for i in range(n):
                permutation = self.getNthPermutation(groupedSymbols, i)
                ungrouped = []
                for p in permutation:
                    ungrouped+=[p[0]]*p[1]
                perm.append(tuple(ungrouped))

        return perm
    
    def factorial(self,n):
        """Calculates the factorial of a non-negative integer.

        Args:
            n: A non-negative integer.

        Returns:
            The factorial of n (n!).
        """
        r = 1
        while n > 1:
            r *= n
            n-=1
        return r
    
    def getNthPermutation(self,symbols, n):
        """Generates the nth lexicographical permutation of a list of symbols.

        Args:
            symbols: A list of items (symbols) to permute.
            n: An integer representing the index (0-based) of the
               desired permutation.

        Returns:
            A tuple representing the nth permutation of the symbols.
        """
        return self.permutation(symbols, self.n_to_factoradic(n))


    def n_to_factoradic(self,n, p = 2):
        """Converts a non-negative integer to its factoradic representation.

        The factoradic representation is a sequence of digits where the
        ith digit (from the right, 0-indexed) is less than or equal to i.
        It's used in algorithms for generating permutations.

        Args:
            n: A non-negative integer to convert.
            p: An integer used internally for recursion (default is 2).

        Returns:
            A list of integers representing the factoradic of n.
        """
        if n < p:
            return [n]
        ret = self.n_to_factoradic((n // p) | 0, p + 1)
        ret.append(n % p)
        return ret
    
    
    def permutation(self, symbols, factoradic):
        """Generates a permutation of symbols based on its factoradic representation.

        Args:
            symbols: A list of items (symbols) to permute.
            factoradic: A list of integers representing the factoradic
                        of the desired permutation index.

        Returns:
            A tuple representing the permutation corresponding to the
            given factoradic.
        """
        factoradic.append(0)
        while len(factoradic) < len(symbols):
            factoradic = [0] + factoradic
        ret = []
        s = symbols[:]
        while len(factoradic)>0:
            f = factoradic[0]
            del factoradic[0]
            ret.append(s[f])
            del s[f]
            
        return tuple(ret)

