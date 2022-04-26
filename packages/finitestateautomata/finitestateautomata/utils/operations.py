# operations on finite state automata

OP_FSA_ACCEPTS = 'accepts'
OP_FSA_IS_DETERMINISTIC = 'isDeterministic'
OP_FSA_AS_DFA = 'asDFA', 
OP_FSA_ELIMINATE_EPSILON = 'eliminateEpsilon', 
OP_FSA_ALPHABET = 'alphabet', 
OP_FSA_COMPLETE = 'complete', 
OP_FSA_COMPLEMENT = 'complement', 
OP_FSA_PRODUCT = 'product', 
OP_FSA_STRICT_PRODUCT = 'strictProduct', 
OP_FSA_PRODUCT_BUCHI = 'productBuchi', 
OP_FSA_REACHABLE_STATES = 'reachableStates', 
OP_FSA_STRICT_PRODUCT_BUCHI = 'strictProductBuchi', 
OP_FSA_LANGUAGE_EMPTY = 'languageEmpty', 
OP_FSA_LANGUAGE_EMPTY_BUCHI = 'languageEmptyBuchi', 
OP_FSA_LANGUAGE_INCLUDED = 'languageIncluded', 
OP_FSA_MINIMIZE = 'minimize', 
OP_FSA_MINIMIZE_BUCHI = 'minimizeBuchi', 
OP_FSA_RELABEL = 'relabel', 
OP_FSA_AS_REGEX = 'asRegEx'

OP_REGEX_CONVERT_FSA = 'convertRegEx'
OP_REGEX_CONVERT_OMEGA_REGEX = 'convertOmegaRegEx'

OP_LTL_CONVERT_LTL = 'convertLTL'

AutomataOperations = [OP_FSA_ACCEPTS, OP_FSA_IS_DETERMINISTIC, OP_FSA_AS_DFA, OP_FSA_ELIMINATE_EPSILON, OP_FSA_ALPHABET, OP_FSA_COMPLETE, OP_FSA_COMPLEMENT, OP_FSA_PRODUCT, OP_FSA_STRICT_PRODUCT, OP_FSA_PRODUCT_BUCHI, OP_FSA_STRICT_PRODUCT_BUCHI, OP_FSA_REACHABLE_STATES, OP_FSA_LANGUAGE_EMPTY, OP_FSA_LANGUAGE_EMPTY_BUCHI, OP_FSA_LANGUAGE_INCLUDED, OP_FSA_MINIMIZE, OP_FSA_MINIMIZE_BUCHI, OP_FSA_RELABEL, OP_FSA_AS_REGEX ]

# FSA Transformations

OP_FSA_AS_DFA_NAME = lambda name: name + "_DFA"
OP_FSA_ELIMINATE_EPSILON_NAME = lambda name: name + "_no_eps"
OP_FSA_COMPLETE_NAME = lambda name: name + "_complete"
OP_FSA_COMPLEMENT_NAME = lambda name: name + "_complement"
OP_FSA_PRODUCT_NAME = lambda name, nameB: name + "_" + nameB + "_prod"
OP_FSA_STRICT_PRODUCT_NAME = lambda name, nameB: name + "_" + nameB + "_sprod"
OP_FSA_PRODUCT_BUCHI_NAME = lambda name, nameB: name + "_" + nameB + "_prodnba"
OP_FSA_PRODUCT_STRICT_BUCHI_NAME = lambda name, nameB: name + "_" + nameB + "_prodgnba"
OP_FSA_MINIMIZE_NAME = lambda name: name + "_min"
OP_FSA_MINIMIZE_BUCHI_NAME = lambda name: name + "_minB"
OP_FSA_RELABEL_NAME = lambda name: name + "_relabeled"
OP_FSA_AS_REGEX_NAME = lambda name: name + "_RegEx"
OP_LTL_CONVERT_LTL_NAME = lambda name: name + "_ltl"


RegExOperations = [OP_REGEX_CONVERT_FSA, OP_REGEX_CONVERT_OMEGA_REGEX]

LTLOperations = ['convertLTL']

OtherOperations = []

Operations = AutomataOperations + RegExOperations + LTLOperations + OtherOperations

OperationDescriptions = [
    OP_FSA_ACCEPTS + ' (checks if provided word is accepted, requires input word)',
    OP_FSA_IS_DETERMINISTIC + ' (check if automaton is deterministic)',
    OP_FSA_AS_DFA + ' (convert FSA to DFA)', 
    OP_FSA_ELIMINATE_EPSILON + ' (eliminate epsilon transitions from FSA)', 
    OP_FSA_ALPHABET + ' (determine the alphabet of the FSA)', 
    OP_FSA_COMPLETE + ' (make the FSA complete)', 
    OP_FSA_COMPLEMENT + ' (determine the complement FSA)', 
    OP_FSA_PRODUCT + ' (determine product with secondary automaton)', 
    OP_FSA_STRICT_PRODUCT + ' (determine strict product with secondary automaton)', 
    OP_FSA_PRODUCT_BUCHI + ' (determine product of a Buchi automaton with secondary Buchi automaton)', 
    OP_FSA_STRICT_PRODUCT_BUCHI + '(determine the strict product of a Buchi automaton with secondary Buchi automaton)', 
    OP_FSA_REACHABLE_STATES + ' (determine the reachable states of the automaton)', 
    OP_FSA_LANGUAGE_EMPTY + ' (check if the language of the automaton is empty)', 
    OP_FSA_LANGUAGE_EMPTY_BUCHI + ' (check if the language of the Buchi automaton is empty)', 
    OP_FSA_LANGUAGE_INCLUDED + ' (check if the language is included in the language of the secondary automaton)', 
    OP_FSA_MINIMIZE + ' (minimize the automaton)', 
    OP_FSA_MINIMIZE_BUCHI + ' (minimize the Buchi automaton)', 
    OP_FSA_RELABEL + ' (relabel the states of the automaton)', 
    OP_FSA_AS_REGEX + ' (convert the automaton to a regular expression)',
    OP_REGEX_CONVERT_FSA + ' (convert the regular expresson to an FSA)',
    OP_REGEX_CONVERT_OMEGA_REGEX + ' (convert the omega-regular expression to a Buchi automaton)',
    OP_LTL_CONVERT_LTL + ' (convert the LTL formula to an automaton)'
]

