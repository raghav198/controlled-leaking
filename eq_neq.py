from coyote.disjoint_set import DisjointSet
from typing import TypeVar

class UnsatisfiableConstraint(Exception): ...

T = TypeVar('T')
class EqNeq(DisjointSet[T]):
    def __init__(self):
        super().__init__()
        self.uneq: set[tuple[T, T]] = set()
        
    def require_neq(self, item1: T, item2: T):
        if self.contains(item1) and self.contains(item2) and self.is_equivalent(item1, item2):
            raise UnsatisfiableConstraint(f'{item1} and {item2} are already equal!')
        if not self.contains(item1):
            self.add(item1)
        if not self.contains(item2):
            self.add(item2)
            
        self.uneq.add((item1, item2))
        
    def union(self, item1: T, item2: T):
        if self.is_unequal(item1, item2):
            raise UnsatisfiableConstraint(f'{item1} and {item2} are required to be unequal!')
        return super().union(item1, item2)
    
    def is_unequal(self, item1: T, item2: T):
        return (item1, item2) in self.uneq or (item2, item1) in self.uneq