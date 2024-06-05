from sortedcontainers import SortedSet

class ConjunctiveNormalForm(SortedSet):
    '''Representing a conjunction of formulas as sorted set of formulas.'''
    def __init__(self, iterable=None, key=None):
        super().__init__(iterable)

    def __hash__(self)->int:
        result: int = 1
        for el in self:
            result = hash((result, el))
        return result

    def __eq__(self, other: 'ConjunctiveNormalForm')->bool:
        assert isinstance(other, ConjunctiveNormalForm)
        if len(self) != len(other):
            return False
        if hash(self) != hash(other):
            return False
        for si, oi in zip(self, other):
            if not si.__eq__(oi):
                return False
        return True

    def __lt__(self, other: 'ConjunctiveNormalForm')->bool:
        assert isinstance(other, ConjunctiveNormalForm)
        ls = len(self)
        lo = len(other)
        if ls != lo:
            return ls<lo
        hs = hash(self)
        ho = hash(other)
        if hs != ho:
            return hs<ho
        for si, oi in zip(self, other):
            if si.__lt__(oi):
                return True
            if oi.__lt__(si):
                return False
        return False

    def logical_and(self, cnf: 'ConjunctiveNormalForm') -> \
            'ConjunctiveNormalForm':
        result = ConjunctiveNormalForm()
        result.update(self)
        result.update(cnf)
        return result

class DisjunctiveNormalForm(SortedSet):
    '''Representing a disjunction of conjunctive formulas as sorted set of ConjunctiveNormalForm.
    Note that it should contain only objects of type ConjunctiveNormalForm'''
    def __init__(self, iterable=None, key=None):
        super().__init__(iterable)

    def __hash__(self)->int:
        return hash(frozenset(self))

    def __eq__(self, other: 'DisjunctiveNormalForm')->bool:
        assert isinstance(other, DisjunctiveNormalForm)
        if len(self) != len(other):
            return False
        if hash(self) != hash(other):
            return False
        for si, oi in zip(self, other):
            if not si.__eq__(oi):
                return False
        return True

    def __lt__(self, other: 'DisjunctiveNormalForm')->bool:
        assert isinstance(other, DisjunctiveNormalForm)
        ls = len(self)
        lo = len(other)
        if ls != lo:
            return ls<lo
        hs = hash(self)
        ho = hash(other)
        if hs != ho:
            return hs<ho
        for si, oi in zip(self, other):
            if si.__lt__(oi):
                return True
            if oi.__lt__(si):
                return False
        return False

    def logical_and(self, dnf: 'DisjunctiveNormalForm') -> \
        'DisjunctiveNormalForm':
        '''Perform logical and operation on two formulas in set disjunctive normal form.'''
        result:DisjunctiveNormalForm = DisjunctiveNormalForm()
        for dt1 in self:
            for dt2 in dnf:
                nt: ConjunctiveNormalForm = ConjunctiveNormalForm()
                nt.update(dt1)
                nt.update(dt2)
                result.add(nt)
        return result

    def logical_or(self, dnf: 'DisjunctiveNormalForm') -> \
         'DisjunctiveNormalForm':
        '''Perform logical or operation on two formulas in set disjunctive normal form.'''
        return self.union(dnf)
