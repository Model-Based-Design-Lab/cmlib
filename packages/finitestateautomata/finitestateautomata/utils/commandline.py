
'''Operations on finite state automata '''

import argparse
import sys
from typing import Optional, Tuple
from finitestateautomata.libfsa import Automaton, FSAException
from finitestateautomata.libregex import RegEx,RegExTerm
from finitestateautomata.libltl import LTLFormula
from finitestateautomata.utils.utils import printStates, printSetOfStates
from finitestateautomata.utils.operations import Operations, \
    AutomataOperations, LTLOperations, OperationDescriptions, RegExOperations, \
    OP_FSA_ACCEPTS, OP_FSA_ALPHABET, OP_FSA_AS_DFA, OP_FSA_AS_DFA_NAME, \
    OP_FSA_AS_REGEX, OP_FSA_AS_REGEX_NAME, OP_FSA_COMPLEMENT, \
    OP_FSA_COMPLEMENT_NAME, OP_FSA_COMPLETE, OP_FSA_COMPLETE_NAME, \
    OP_FSA_ELIMINATE_EPSILON, OP_FSA_ELIMINATE_EPSILON_NAME, \
    OP_FSA_IS_DETERMINISTIC, OP_FSA_LANGUAGE_EMPTY, \
    OP_FSA_LANGUAGE_EMPTY_BUCHI, OP_FSA_LANGUAGE_INCLUDED, \
    OP_FSA_MINIMIZE, OP_FSA_MINIMIZE_BUCHI, OP_FSA_MINIMIZE_BUCHI_NAME, \
    OP_FSA_MINIMIZE_NAME, OP_FSA_PRODUCT, OP_FSA_PRODUCT_BUCHI, \
    OP_FSA_PRODUCT_BUCHI_NAME, OP_FSA_PRODUCT_NAME, \
    OP_FSA_PRODUCT_STRICT_BUCHI_NAME, OP_FSA_REACHABLE_STATES, \
    OP_FSA_RELABEL, OP_FSA_RELABEL_NAME, OP_FSA_STRICT_PRODUCT, \
    OP_FSA_STRICT_PRODUCT_BUCHI, OP_FSA_STRICT_PRODUCT_NAME, \
    OP_LTL_CONVERT_LTL, OP_LTL_CONVERT_LTL_NAME, OP_REGEX_CONVERT_FSA, \
    OP_REGEX_CONVERT_OMEGA_REGEX

def main():
    ''' the main entry point '''

    # optional help flag explaining usage of each individual operation
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('-oh', '--operationhelp', dest='opHelp', nargs="?", const=" ")
    options, _ = parser.parse_known_args() # Only use options of parser above

    if options.opHelp: # Check if -oh has been called
        if options.opHelp not in Operations:
            print(f"Operation '{options.opHelp}' does not exist. List of operations:\n\t- { \
                "\n\t- ".join(Operations)}")
        else:
            print(f"{options.opHelp}: {OperationDescriptions[Operations.index(options.opHelp)]}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Perform operations on finite state automata.\n' \
            'https://computationalmodeling.info')
    parser.add_argument('automaton_or_regularexpression', \
                        help="the automaton or regular expression to analyze")
    parser.add_argument('-sa', '--secondaryautomaton', dest='secondaryAutomaton',
                        help="a secondary automaton for the operation")
    parser.add_argument('-op', '--operation', dest='operation',
            help=f"the operation or analysis to perform, one of: {"; \n".join(Operations) \
            }.\n" \
            "Use 'finitestateautomata -oh OPERATION' for information about the specific operation.")
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
        sys.stderr.write(f"Unknown operation: {args.operation}\n")
        sys.stderr.write(f"Operation should be one of: {", ".join(Operations)}.\n")
        sys.exit(1)

    dsl = None
    if args.automaton_or_regularexpression:
        try:
            with open(args.automaton_or_regularexpression, 'r', encoding='utf-8') as fsa_file:
                dsl = fsa_file.read()
        except FileNotFoundError:
            sys.stderr.write(f"File does not exist: {args.automaton_or_regularexpression}.\n")
            sys.exit(1)

    try:
        process(args, dsl)
    except FSAException as e:
        sys.stderr.write(f"{e}\n")
        # raise e
        sys.exit(1)

    sys.exit(0)


def require_secondary_automaton(args)->Tuple[str,Automaton]:
    ''' ensure a secondary automaton was specified on the command line and return it '''
    if args.secondaryAutomaton is None:
        raise FSAException("A secondary automaton must be specified.")
    try:
        with open(args.secondaryAutomaton, 'r', encoding='utf-8') as fsa_file:
            sa_dsl = fsa_file.read()
        name_b, b = Automaton.from_dsl(sa_dsl)
        return name_b, b
    except FileNotFoundError:
        raise FSAException(f"Failed to read file: {args.secondaryAutomaton}.\n") # pylint: disable=raise-missing-from


def require_automaton(a:Optional[Automaton])->Automaton:
    ''' ensure an automaton was specified on the command line and return it '''
    if a is None:
        raise FSAException("Automaton is not defined")
    return a

def require_reg_ex(r: Optional[RegEx])->RegEx:
    ''' ensure a regular expression was specified on the command line and return it '''
    if r is None:
        raise FSAException("Regular expression is not defined")
    return r

def require_ltl_formula(f:Optional[LTLFormula])->LTLFormula:
    ''' ensure an LTL formula was specified on the command line and return it '''
    if f is None:
        raise FSAException("LTL Formula is not defined")
    return f

def process(args, dsl):
    ''' process the specified command '''
    a = None
    r = None
    f = None
    name = None

    if args.operation in AutomataOperations:
        name, a = Automaton.from_dsl(dsl)

    if args.operation in RegExOperations:
        name, r = RegEx.fromDSL(dsl)

    if args.operation in LTLOperations:
        name, f = LTLFormula.fromDSL(dsl)

    if args.operation not in Operations:
        print("Unknown operation or no operation provided")
        print(f"Operation should be one of: {", ".join(Operations)}.")
        sys.exit(1)

    # accepts (requires input word)
    if args.operation == OP_FSA_ACCEPTS:
        res, path = require_automaton(a).accepts_with_path(args.inputtrace)
        print(res)
        if res and path is not None:
            printStates(path)

    # isDeterministic
    if args.operation == OP_FSA_IS_DETERMINISTIC:
        print(require_automaton(a).is_deterministic())

    # asDFA
    if args.operation == OP_FSA_AS_DFA:
        res = require_automaton(a).as_dfa()
        print(res.as_dsl(OP_FSA_AS_DFA_NAME(name)))

    # eliminateEpsilon
    if args.operation == OP_FSA_ELIMINATE_EPSILON:
        res = require_automaton(a).eliminate_epsilon_transitions()
        print(res.as_dsl(OP_FSA_ELIMINATE_EPSILON_NAME(name)))

    # alphabet
    if args.operation == OP_FSA_ALPHABET:
        print(require_automaton(a).alphabet())

    # complete
    if args.operation == OP_FSA_COMPLETE:
        res = require_automaton(a).complete()
        print(res.as_dsl(OP_FSA_COMPLETE_NAME(name)))

    # complement
    if args.operation == OP_FSA_COMPLEMENT:
        res = require_automaton(a).complement()
        print(res.as_dsl(OP_FSA_COMPLEMENT_NAME(name)))

    # product (requires secondary automaton)
    if args.operation == OP_FSA_PRODUCT:
        name_b, b = require_secondary_automaton(args)
        res = require_automaton(a).product(b)
        print(res.as_dsl(OP_FSA_PRODUCT_NAME(name, name_b)))

    # strictProduct (requires secondary automaton)
    if args.operation == OP_FSA_STRICT_PRODUCT:
        name_b, b = require_secondary_automaton(args)
        res = require_automaton(a).strict_product(b)
        print(res.as_dsl(OP_FSA_STRICT_PRODUCT_NAME(name, name_b)))

    # productBuchi (requires secondary automaton)
    if args.operation == OP_FSA_PRODUCT_BUCHI:
        name_b, b = require_secondary_automaton(args)
        res = require_automaton(a).as_regular_buchi_automaton() \
            .product_buchi(b.as_regular_buchi_automaton())
        print(res.as_dsl(OP_FSA_PRODUCT_BUCHI_NAME(name, name_b)))

    # strictProductBuchi (requires secondary automaton)
    if args.operation == OP_FSA_STRICT_PRODUCT_BUCHI:
        name_b, b = require_secondary_automaton(args)
        res = require_automaton(a).strict_product_buchi(b)
        print(res.as_dsl(OP_FSA_PRODUCT_STRICT_BUCHI_NAME(name, name_b)))

    # languageEmpty
    if args.operation == OP_FSA_LANGUAGE_EMPTY:
        empty, word, path = require_automaton(a).language_empty()
        print(empty)
        if not empty:
            print(word)
            if path is not None:
                printStates(path)

    # languageEmptyBuchi
    if args.operation == OP_FSA_LANGUAGE_EMPTY_BUCHI:
        r_a, state_map = require_automaton(a).as_regular_buchi_automaton_with_state_map()
        empty, wordprefix, wordrepeat, pathprefix, pathrepeat = r_a.language_empty_buchi()
        print(empty)
        if not empty:
            print(wordprefix)
            print(wordrepeat)
            if pathprefix is not None and pathrepeat is not None:
                printStates(list(map(lambda s: state_map[s], pathprefix)))
                printStates(list(map(lambda s: state_map[s], pathrepeat)))

    # reachable states
    if args.operation == OP_FSA_REACHABLE_STATES:
        res = require_automaton(a).reachable_states()
        printSetOfStates(res)

    # languageIncluded
    if args.operation == OP_FSA_LANGUAGE_INCLUDED:
        _, b = require_secondary_automaton(args)
        included, word = require_automaton(a).language_included(b)
        print(included)
        if not included:
            print(word)

    # minimize
    if args.operation == OP_FSA_MINIMIZE:
        res = require_automaton(a).minimize()
        print(res.as_dsl(OP_FSA_MINIMIZE_NAME(name)))

    # minimize Büchi
    if args.operation == OP_FSA_MINIMIZE_BUCHI:
        res = require_automaton(a).minimize_buchi()
        print(res.as_dsl(OP_FSA_MINIMIZE_BUCHI_NAME(name)))

    # relabel
    if args.operation == OP_FSA_RELABEL:
        res = require_automaton(a).relabel_states()
        print(res.as_dsl(OP_FSA_RELABEL_NAME(name)))

    # convertRegEx (requires regular expression)
    if args.operation == OP_REGEX_CONVERT_FSA:
        res = require_reg_ex(r).asFSA()
        print(res.asDSL(name))  # type: ignore

    # convertOmegaRegEx (requires omega-regular expression)
    if args.operation == OP_REGEX_CONVERT_OMEGA_REGEX:
        res = require_reg_ex(r).asNBA()
        print(res.asDSL(name))  # type: ignore

    # convertLTL (requires ltl formula)
    if args.operation == OP_LTL_CONVERT_LTL:
        res = require_ltl_formula(f).asFSA()
        print(res.as_dsl(OP_LTL_CONVERT_LTL_NAME(name)))

    # asRegEx
    if args.operation == OP_FSA_AS_REGEX:
        res = RegEx.fromFSA(a, OP_FSA_AS_REGEX_NAME(name))
        print(res.asDSL(OP_FSA_AS_REGEX_NAME(name)))

    # # test parser
    # if args.operation == 'testLTLParser':
    #     print(F.asDSL('name'))

if __name__ == "__main__":
    main()
