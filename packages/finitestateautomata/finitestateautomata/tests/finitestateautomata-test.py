from unittest import TestCase
from finitestateautomata.libfsa import Automaton

class TestFiniteStateAutomata(TestCase):

    def test_automata(self):
        """ to add some tests """
        A = Automaton()
        A.addState("A")
        A.addState("B")
        A.makeInitialState("A")
        A.makeInitialState("B")
        A.makeFinalState("B")

        A.addTransition("A", "a", "B")
        A.addTransition("A", "b", "B")
        A.addTransition("B", "c", "A")
        A.addTransition("B", "d", "B")

        print(A.accepts("a,d,c,b"))
        print(A.accepts("a,b,c"))

        print(A.isDeterministic())

        print(A)

        B = A.asDFA()

        print(B)

        print(B.isDeterministic())

        C = B.relabelStates()

        print(C)

        dsl = C.asDSL("myFSA")

        dsl = "finite state automaton myFSA {\n        S2 final -- d --> S2 final i\n        S2 -- c --> S3\n        S1 initial f -- c --> S3\n        S1 -- a, d, b --> S2\n        S3 -- a, b --> S2\n}"

        print(dsl)

        (name, D) = Automaton.fromDSL(dsl)

        print(D)
        print(D.asDSL(name))

   

