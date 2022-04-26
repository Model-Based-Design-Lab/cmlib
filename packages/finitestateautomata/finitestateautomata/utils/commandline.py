
'''Operations on finite state automata '''

import argparse
from finitestateautomata.libfsa import Automaton
from finitestateautomata.libregex import RegEx
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
                        help="the reguar expression to use")
    parser.add_argument('-ltl', '--ltlformula', dest='ltlFormula',
                        help="the LTL formula to use")
    parser.add_argument('-w', '--word', dest='inputtrace',
                        help="an input word, a comma-separated list of symbols")

    args = parser.parse_args()

    if args.operation not in Operations:
        sys.stderr.write("Unknown operation: {}\n".format(args.operation))
        sys.stderr.write("Operation should be one of: {}.\n".format(", ".join(Operations)))
        exit(1)

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
        sys.stderr.write("An error occurred: {}\n".format(e))
        # raise e
        exit(1)

    exit(0)


def requireSecondaryAutomaton(args):
    if args.secondaryAutomaton is None:
        raise Exception("A secondary automaton must be specified.")
    try:
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        nameB, B = Automaton.fromDSL(saDsl)
        return nameB, B
    except FileNotFoundError as e:
        sys.stderr.write("Failed to read file: {}.\n".format(args.secondaryAutomaton))
        
        


def process(args, dsl):

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
        res, path = A.acceptsWithPath(args.inputtrace)
        print(res)
        if res:
            printStates(path)

    # isDeterministic
    if args.operation == OP_FSA_IS_DETERMINISTIC:
        print(A.isDeterministic())

    # asDFA
    if args.operation == OP_FSA_AS_DFA:
        res = A.asDFA()
        print(res.asDSL(OP_FSA_AS_DFA_NAME(name)))

    # eliminateEpsilon
    if args.operation == OP_FSA_ELIMINATE_EPSILON:
        res = A.eliminateEpsilonTransitions()
        print(res.asDSL(OP_FSA_ELIMINATE_EPSILON_NAME(name)))

    # alphabet
    if args.operation == OP_FSA_ALPHABET:
        print(A.alphabet())

    # complete
    if args.operation == OP_FSA_COMPLETE:
        res = A.complete()
        print(res.asDSL(OP_FSA_COMPLETE_NAME(name)))

    # complement
    if args.operation == OP_FSA_COMPLEMENT:
        res = A.complement()
        print(res.asDSL(OP_FSA_COMPLEMENT_NAME(name)))

    # product (requires secondary automaton)
    if args.operation == OP_FSA_PRODUCT:
        nameB, B = requireSecondaryAutomaton(args)
        res = A.product(B)
        print(res.asDSL(OP_FSA_PRODUCT_NAME(name, nameB)))

    # strictProduct (requires secondary automaton)
    if args.operation == OP_FSA_STRICT_PRODUCT:
        nameB, B = requireSecondaryAutomaton(args)
        res = A.strictProduct(B)
        print(res.asDSL(OP_FSA_STRICT_PRODUCT_NAME(name, nameB)))

    # productBuchi (requires secondary automaton)
    if args.operation == OP_FSA_PRODUCT_BUCHI:
        nameB, B = requireSecondaryAutomaton(args)
        res = A.asRegularBuchiAutomaton().productBuchi(B.asRegularBuchiAutomaton())
        print(res.asDSL(OP_FSA_PRODUCT_BUCHI_NAME(name, nameB)))

    # strictProductBuchi (requires secondary automaton)
    if args.operation == OP_FSA_STRICT_PRODUCT_BUCHI:
        nameB, B = requireSecondaryAutomaton(args)
        res = A.strictProductBuchi(B)
        print(res.asDSL(OP_FSA_PRODUCT_STRICT_BUCHI_NAME(name, nameB)))

    # languageEmpty
    if args.operation == OP_FSA_LANGUAGE_EMPTY:
        empty, word, path = A.languageEmpty()
        print(empty)
        if not empty:
            print(word)
            printStates(path)

    # languageEmptyBuchi
    if args.operation == OP_FSA_LANGUAGE_EMPTY_BUCHI:
        empty, wordprefix, wordrepeat, pathprefix, pathrepeat = A.asRegularBuchiAutomaton().languageEmptyBuchiAlternative()
        print(empty)
        if not empty:
            print(wordprefix)
            print(wordrepeat)
            printStates(pathprefix)
            printStates(pathrepeat)

    # languageEmptyBuchi
    if args.operation == OP_FSA_REACHABLE_STATES:
        res = A.reachableStates()
        printSetOfStates(res)

    # languageIncluded
    if args.operation == OP_FSA_LANGUAGE_INCLUDED:
        _, B = requireSecondaryAutomaton(args)
        included, word = A.languageIncluded(B)
        print(included)
        if not included:
            print(word)

    # minimize
    if args.operation == OP_FSA_MINIMIZE:
        res = A.minimize()
        print(res.asDSL(OP_FSA_MINIMIZE_NAME(name)))

    # minimize Büchi
    if args.operation == OP_FSA_MINIMIZE_BUCHI:
        res = A.minimizeBuchi()
        print(res.asDSL(OP_FSA_MINIMIZE_BUCHI_NAME(name)))

    # relabel
    if args.operation == OP_FSA_RELABEL:
        res = A.relabelStates()
        print(res.asDSL(OP_FSA_RELABEL_NAME(name)))

    # convertRegEx (requires regular expression)
    if args.operation == OP_REGEX_CONVERT_FSA:
        res = R.asFSA()
        print(res.asDSL(name))

    # convertOmegaRegEx (requires omega-regular expression)
    if args.operation == OP_REGEX_CONVERT_OMEGA_REGEX:
        res = R.asNBA()
        print(res.asDSL(name))

    # convertLTL (requires ltl formula)
    if args.operation == OP_LTL_CONVERT_LTL:
        res = F.asFSA()
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
