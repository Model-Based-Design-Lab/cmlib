
'''Operations on finite state automata '''

import argparse
from typing import Optional, Tuple
from finitestateautomata.libfsa import Automaton
from finitestateautomata.libregex import RegEx,RegExTerm
from finitestateautomata.libltl import LTLFormula
import sys
from finitestateautomata.utils.utils import printStates, printSetOfStates
from finitestateautomata.utils.operations import Operations, AutomataOperations, LTLOperations, OperationDescriptions, RegExOperations, OP_FSA_ACCEPTS, OP_FSA_ALPHABET, OP_FSA_AS_DFA, OP_FSA_AS_DFA_NAME, OP_FSA_AS_REGEX, OP_FSA_AS_REGEX_NAME, OP_FSA_COMPLEMENT, OP_FSA_COMPLEMENT_NAME, OP_FSA_COMPLETE, OP_FSA_COMPLETE_NAME, OP_FSA_ELIMINATE_EPSILON, OP_FSA_ELIMINATE_EPSILON_NAME, OP_FSA_IS_DETERMINISTIC, OP_FSA_LANGUAGE_EMPTY, OP_FSA_LANGUAGE_EMPTY_BUCHI, OP_FSA_LANGUAGE_INCLUDED, OP_FSA_MINIMIZE, OP_FSA_MINIMIZE_BUCHI, OP_FSA_MINIMIZE_BUCHI_NAME, OP_FSA_MINIMIZE_NAME, OP_FSA_PRODUCT, OP_FSA_PRODUCT_BUCHI, OP_FSA_PRODUCT_BUCHI_NAME, OP_FSA_PRODUCT_NAME, OP_FSA_PRODUCT_STRICT_BUCHI_NAME, OP_FSA_REACHABLE_STATES, OP_FSA_RELABEL, OP_FSA_RELABEL_NAME, OP_FSA_STRICT_PRODUCT, OP_FSA_STRICT_PRODUCT_BUCHI, OP_FSA_STRICT_PRODUCT_NAME, OP_LTL_CONVERT_LTL, OP_LTL_CONVERT_LTL_NAME, OP_REGEX_CONVERT_FSA, OP_REGEX_CONVERT_OMEGA_REGEX

def main():

    parser = argparse.ArgumentParser(
        description='Perform operations on finite state automata.\nhttps://computationalmodeling.info')
    parser.add_argument('automaton_or_regularexpression', help="the automaton or regular expression to analyze")
    parser.add_argument('-sa', '--secondaryautomaton', dest='secondaryAutomaton',
                        help="a secondary automaton for the operation")
    parser.add_argument('-op', '--operation', dest='operation',
                        help="the operation or analysis to perform, one of : {}".format("; ".join(OperationDescriptions)))
    parser.add_argument('-oa', '--outputautomaton', dest='outputAutomaton',
                        help="the outputfile to write output automata to")
    parser.add_argument('-re', '--regularexpression', dest='regularExpression',
                        help="the regular expression to use")
    parser.add_argument('-ltl', '--ltlformula', dest='ltlFormula',
                        help="the LTL formula to use")
    parser.add_argument('-w', '--word', dest='inputtrace',
                        help="an input word, a comma-separated list of symbols")

    args = parser.parse_args()

    if args.operation not in Operations:
        sys.stderr.write("Unknown operation: {}\n".format(args.operation))
        sys.stderr.write("Operation should be one of: {}.\n".format(", ".join(Operations)))
        exit(1)

    dsl = None
    if args.automaton_or_regularexpression:
        try:
            with open(args.automaton_or_regularexpression, 'r') as fsaFile:
                dsl = fsaFile.read()
        except FileNotFoundError as e:
            sys.stderr.write("File does not exist: {}.\n".format(args.automaton_or_regularexpression))
            exit(1)

    try:
        process(args, dsl)
    except Exception as e:
        sys.stderr.write("{}\n".format(e))
        # raise e
        exit(1)

    exit(0)


def requireSecondaryAutomaton(args)->Tuple[str,Automaton]:
    if args.secondaryAutomaton is None:
        raise Exception("A secondary automaton must be specified.")
    try:
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        nameB, B = Automaton.fromDSL(saDsl)
        return nameB, B
    except FileNotFoundError as e:
        raise Exception("Failed to read file: {}.\n".format(args.secondaryAutomaton))
        
        
def requireAutomaton(A:Optional[Automaton])->Automaton:
    if A is None:
        raise Exception("Automaton is not defined")
    return A

def requireRegEx(R: Optional[RegExTerm])->RegExTerm:
    if R is None:
        raise Exception("Regular expression is not defined")
    return R

def requireLTLFormula(F:Optional[LTLFormula])->LTLFormula:
    if F is None:
        raise Exception("LTL Formula is not defined")
    return F

def process(args, dsl):

    A = None
    R = None
    F = None
    name = None

    if args.operation in AutomataOperations:
        name, A = Automaton.fromDSL(dsl)

    if args.operation in RegExOperations:
        name, R = RegEx.fromDSL(dsl)

    if args.operation in LTLOperations:
        name, F = LTLFormula.fromDSL(dsl)

    if args.operation not in Operations:
        print("Unknown operation or no operation provided")
        print("Operation should be one of: {}.".format(", ".join(Operations)))
        exit(1)

    # accepts (requires input word)
    if args.operation == OP_FSA_ACCEPTS:
        res, path = requireAutomaton(A).acceptsWithPath(args.inputtrace)
        print(res)
        if res and path is not None:
            printStates(path)

    # isDeterministic
    if args.operation == OP_FSA_IS_DETERMINISTIC:
        print(requireAutomaton(A).isDeterministic())

    # asDFA
    if args.operation == OP_FSA_AS_DFA:
        res = requireAutomaton(A).asDFA()
        print(res.asDSL(OP_FSA_AS_DFA_NAME(name)))

    # eliminateEpsilon
    if args.operation == OP_FSA_ELIMINATE_EPSILON:
        res = requireAutomaton(A).eliminateEpsilonTransitions()
        print(res.asDSL(OP_FSA_ELIMINATE_EPSILON_NAME(name)))

    # alphabet
    if args.operation == OP_FSA_ALPHABET:
        print(requireAutomaton(A).alphabet())

    # complete
    if args.operation == OP_FSA_COMPLETE:
        res = requireAutomaton(A).complete()
        print(res.asDSL(OP_FSA_COMPLETE_NAME(name)))

    # complement
    if args.operation == OP_FSA_COMPLEMENT:
        res = requireAutomaton(A).complement()
        print(res.asDSL(OP_FSA_COMPLEMENT_NAME(name)))

    # product (requires secondary automaton)
    if args.operation == OP_FSA_PRODUCT:
        nameB, B = requireSecondaryAutomaton(args)
        res = requireAutomaton(A).product(B)
        print(res.asDSL(OP_FSA_PRODUCT_NAME(name, nameB)))

    # strictProduct (requires secondary automaton)
    if args.operation == OP_FSA_STRICT_PRODUCT:
        nameB, B = requireSecondaryAutomaton(args)
        res = requireAutomaton(A).strictProduct(B)
        print(res.asDSL(OP_FSA_STRICT_PRODUCT_NAME(name, nameB)))

    # productBuchi (requires secondary automaton)
    if args.operation == OP_FSA_PRODUCT_BUCHI:
        nameB, B = requireSecondaryAutomaton(args)
        res = requireAutomaton(A).asRegularBuchiAutomaton().productBuchi(B.asRegularBuchiAutomaton())
        print(res.asDSL(OP_FSA_PRODUCT_BUCHI_NAME(name, nameB)))

    # strictProductBuchi (requires secondary automaton)
    if args.operation == OP_FSA_STRICT_PRODUCT_BUCHI:
        nameB, B = requireSecondaryAutomaton(args)
        res = requireAutomaton(A).strictProductBuchi(B)
        print(res.asDSL(OP_FSA_PRODUCT_STRICT_BUCHI_NAME(name, nameB)))

    # languageEmpty
    if args.operation == OP_FSA_LANGUAGE_EMPTY:
        empty, word, path = requireAutomaton(A).languageEmpty()
        print(empty)
        if not empty:
            print(word)
            if path is not None:
                printStates(path)

    # languageEmptyBuchi
    if args.operation == OP_FSA_LANGUAGE_EMPTY_BUCHI:
        RA, stateMap = requireAutomaton(A).asRegularBuchiAutomatonWithStateMap()
        empty, wordprefix, wordrepeat, pathprefix, pathrepeat = RA.languageEmptyBuchi()
        print(empty)
        if not empty:
            print(wordprefix)
            print(wordrepeat)
            if pathprefix is not None and pathrepeat is not None:
                printStates(list(map(lambda s: stateMap[s], pathprefix)))
                printStates(list(map(lambda s: stateMap[s], pathrepeat)))

    # reachable states
    if args.operation == OP_FSA_REACHABLE_STATES:
        res = requireAutomaton(A).reachableStates()
        printSetOfStates(res)

    # languageIncluded
    if args.operation == OP_FSA_LANGUAGE_INCLUDED:
        _, B = requireSecondaryAutomaton(args)
        included, word = requireAutomaton(A).languageIncluded(B)
        print(included)
        if not included:
            print(word)

    # minimize
    if args.operation == OP_FSA_MINIMIZE:
        res = requireAutomaton(A).minimize()
        print(res.asDSL(OP_FSA_MINIMIZE_NAME(name)))

    # minimize Büchi
    if args.operation == OP_FSA_MINIMIZE_BUCHI:
        res = requireAutomaton(A).minimizeBuchi()
        print(res.asDSL(OP_FSA_MINIMIZE_BUCHI_NAME(name)))

    # relabel
    if args.operation == OP_FSA_RELABEL:
        res = requireAutomaton(A).relabelStates()
        print(res.asDSL(OP_FSA_RELABEL_NAME(name)))

    # convertRegEx (requires regular expression)
    if args.operation == OP_REGEX_CONVERT_FSA:
        res = requireRegEx(R).asFSA()
        print(res.asDSL(name))  # type: ignore

    # convertOmegaRegEx (requires omega-regular expression)
    if args.operation == OP_REGEX_CONVERT_OMEGA_REGEX:
        res = requireRegEx(R).asNBA()
        print(res.asDSL(name))  # type: ignore

    # convertLTL (requires ltl formula)
    if args.operation == OP_LTL_CONVERT_LTL:
        res = requireLTLFormula(F).asFSA()
        print(res.asDSL(OP_LTL_CONVERT_LTL_NAME(name)))

    # asRegEx
    if args.operation == OP_FSA_AS_REGEX:
        res = RegEx.fromFSA(A, OP_FSA_AS_REGEX_NAME(name))
        print(res.asDSL(OP_FSA_AS_REGEX_NAME(name)))

    # # test parser
    # if args.operation == 'testLTLParser':
    #     print(F.asDSL('name'))

if __name__ == "__main__":
    main()
