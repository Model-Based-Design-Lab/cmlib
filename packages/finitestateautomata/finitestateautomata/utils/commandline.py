
'''Operations on finite state automata '''

import argparse
from finitestateautomata.libfsa import Automaton
from finitestateautomata.libregex import RegEx
from finitestateautomata.libltl import LTLFormula
import sys
from finitestateautomata.utils.utils import printStates, printSetOfStates

AutomataOperations = ['accepts', 'isDeterministic', 'asDFA', 'eliminateEpsilon', 'alphabet', 'complete', 'complement', 'product', 'strictProduct', 'productBuchi', 'reachableStates', 'strictProductBuchi', 'languageEmpty', 'languageEmptyBuchi', 'languageIncluded', 'minimize', 'minimizeBuchi', 'relabel', 'asRegEx']
RegExOperations = ['convertRegEx', 'convertOmegaRegEx']
LTLOperations = ['convertLTL', 'testLTLParser']

OtherOperations = []

operations = AutomataOperations + RegExOperations + LTLOperations + OtherOperations

def main():

    operationDescriptions = [
        "accepts (requires input word)",
        "isDeterministic",
        "asDFA",
        "eliminateEpsilon",
        "alphabet",
        "complete",
        "complement",
        "product (requires secondary automaton)",
        "strictProduct (requires secondary automaton)",
        "productBuchi (requires secondary automaton)",
        "strictProductBuchi (requires secondary automaton)",
        "languageEmpty",
        "languageEmptyBuchi",
        "languageIncluded (requires secondary automaton)",
        "minimize",
        "minimizeBuchi",
        "reachableStates"
        "relabel"
        "convertRegEx (requires regular expression)",
        "convertOmegaRegEx (requires omega-regular expression)",
        "convertLTL (requires ltl formula)",
        "asRegEx"
    ]
    parser = argparse.ArgumentParser(
        description='Perform operations on finite state automata.')
    parser.add_argument('automaton_or_regularexpression', help="the automaton or regular expression to analyze")
    parser.add_argument('-sa', '--secondaryautomaton', dest='secondaryAutomaton',
                        help="a secondary automaton for the operation")
    parser.add_argument('-op', '--operation', dest='operation',
                        help="the operation or analysis to perform, one of : {}".format("; ".join(operationDescriptions)))
    parser.add_argument('-oa', '--outputautomaton', dest='outputAutomaton',
                        help="the outputfile to write output automata to")
    parser.add_argument('-re', '--regularexpression', dest='regularExpression',
                        help="the reguar expression to use")
    parser.add_argument('-ltl', '--ltlformula', dest='ltlFormula',
                        help="the LTL formula to use")
    parser.add_argument('-w', '--word', dest='inputtrace',
                        help="an input word, a comma-separated list of symbols")

    args = parser.parse_args()

    if args.operation not in operations:
        sys.stderr.write("Unknown operation: {}\n".format(args.operation))
        sys.stderr.write("Operation should be one of: {}.\n".format(", ".join(operations)))
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



def process(args, dsl):

    if args.operation in AutomataOperations:
        name, A = Automaton.fromDSL(dsl)

    if args.operation in RegExOperations:
        name, R = RegEx.fromDSL(dsl)

    if args.operation in LTLOperations:
        name, F = LTLFormula.fromDSL(dsl)

    if args.operation not in operations:
        print("Unknown operation or no operation provided")
        print("Operation should be one of: {}.".format(", ".join(operations)))
        exit(1)

    # accepts (requires input word)
    if args.operation == 'accepts':
        res, path = A.acceptsWithPath(args.inputtrace)
        print(res)
        if res:
            printStates(path)

    # isDeterministic
    if args.operation == 'isDeterministic':
        print(A.isDeterministic())

    # asDFA
    if args.operation == 'asDFA':
        res = A.asDFA()
        print(res.asDSL(name+"_DFA"))

    # eliminateEpsilon
    if args.operation == 'eliminateEpsilon':
        res = A.eliminateEpsilonTransitions()
        print(res.asDSL(name+"_no_eps"))

    # alphabet
    if args.operation == 'alphabet':
        print(A.alphabet())

    # complete
    if args.operation == 'complete':
        res = A.complete()
        print(res.asDSL(name+"_complete"))

    # complement
    if args.operation == 'complement':
        res = A.complement()
        print(res.asDSL(name+"_complement"))

    # product (requires secondary automaton)
    if args.operation == 'product':
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        nameB, B = Automaton.fromDSL(saDsl)
        res = A.product(B)
        print(res.asDSL(name + "_" + nameB + "_prod"))

    # strictProduct (requires secondary automaton)
    if args.operation == 'strictProduct':
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        nameB, B = Automaton.fromDSL(saDsl)
        res = A.strictProduct(B)
        print(res.asDSL(name + "_" + nameB + "_sprod"))

    # productBuchi (requires secondary automaton)
    if args.operation == 'productBuchi':
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        nameB, B = Automaton.fromDSL(saDsl)
        res = A.asRegularBuchiAutomaton().productBuchi(B.asRegularBuchiAutomaton())
        print(res.asDSL(name + "_" + nameB + "_prodnba"))

    # strictProductBuchi (requires secondary automaton)
    if args.operation == 'strictProductBuchi':
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        nameB, B = Automaton.fromDSL(saDsl)
        # res = A.asRegularBuchiAutomaton().strictProductBuchi(B.asRegularBuchiAutomaton())
        res = A.strictProductBuchi(B)
        print(res.asDSL(name + "_" + nameB + "_prodgnba"))

    # languageEmpty
    if args.operation == 'languageEmpty':
        empty, word, path = A.languageEmpty()
        print(empty)
        if not empty:
            print(word)
            printStates(path)

    # languageEmptyBuchi
    if args.operation == 'languageEmptyBuchi':
        # empty, wordprefix, wordrepeat = A.asRegularBuchiAutomaton().languageEmptyBuchi()
        empty, wordprefix, wordrepeat, pathprefix, pathrepeat = A.asRegularBuchiAutomaton().languageEmptyBuchiAlternative()
        print(empty)
        if not empty:
            print(wordprefix)
            print(wordrepeat)
            printStates(pathprefix)
            printStates(pathrepeat)

    # languageEmptyBuchi
    if args.operation == 'reachableStates':
        res = A.reachableStates()
        printSetOfStates(res)


    # languageIncluded
    if args.operation == 'languageIncluded':
        with open(args.secondaryAutomaton, 'r') as fsaFile:
            saDsl = fsaFile.read()
        _, B = Automaton.fromDSL(saDsl)
        included, word = A.languageIncluded(B)
        print(included)
        if not included:
            print(word)

    # minimize
    if args.operation == 'minimize':
        res = A.minimize()
        print(res.asDSL(name+"_min"))

    # minimize Büchi
    if args.operation == 'minimizeBuchi':
        res = A.minimizeBuchi()
        print(res.asDSL(name+"_minB"))

    # relabel
    if args.operation == 'relabel':
        res = A.relabelStates()
        print(res.asDSL(name+"_relabeled"))

    # convertRegEx (requires regular expression)
    if args.operation == 'convertRegEx':
        res = R.asFSA()
        print(res.asDSL(name))

    # convertOmegaRegEx (requires omega-regular expression)
    if args.operation == 'convertOmegaRegEx':
        res = R.asNBA()
        print(res.asDSL(name))

    # convertLTL (requires ltl formula)
    if args.operation == 'convertLTL':
        res = F.asFSA()
        print(res.asDSL(name+'_fsa'))

    # asRegEx
    if args.operation == 'asRegEx':
        res = RegEx.fromFSA(A, name+"_RegEx")
        print(res.asDSL(name+"_RegEx"))

    # test parser
    if args.operation == 'testLTLParser':
        print(F.asDSL('name'))

if __name__ == "__main__":
    main()
