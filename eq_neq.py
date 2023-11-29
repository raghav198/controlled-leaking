from __future__ import annotations

from typing import TypeVar

from typing_extensions import override

from coyote.disjoint_set import DisjointSet


class UnsatisfiableConstraint(Exception): ...

T = TypeVar('T')
class EqNeq(DisjointSet[T]):
    """DisjointSet capable of reasoning about inequality constraints
    """
    def __init__(self) -> None:
        super().__init__()
        self.uneq: set[tuple[T, T]] = set()
        
    def require_neq(self, item1: T, item2: T):
        """Force `item1` and `item2` to be in separate equivalence classes"""
        if self.contains(item1) and self.contains(item2) and self.is_equivalent(item1, item2):
            raise UnsatisfiableConstraint(f'{item1} and {item2} are already equal!')
        if not self.contains(item1):
            self.add(item1)
        if not self.contains(item2):
            self.add(item2)
            
        self.uneq.add((item1, item2))
        
    @override
    def copy(self) -> EqNeq[T]:
        new: EqNeq[T] = EqNeq()
        new.vertices.update(self.vertices)
        new.parent.update(self.parent)
        new.children.update(self.children)
        new.uneq.update(self.uneq)
        return new
        
    def union(self, item1: T, item2: T):
        if self.is_unequal(item1, item2):
            raise UnsatisfiableConstraint(f'{item1} and {item2} are required to be unequal!')
        return super().union(item1, item2)
    
    def is_unequal(self, item1: T, item2: T):
        """Are `item1` and `item2` guaranteed to be unequal?"""
        return (item1, item2) in self.uneq or (item2, item1) in self.uneq