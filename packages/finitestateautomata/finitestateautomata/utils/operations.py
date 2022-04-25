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

OP_REGEX_CONVERT_REGEX = 'convertRegEx'
OP_REGEX_CONVERT_OMEGA_REGEX = 'convertOmegaRegEx'

OP_LTL_CONVERT_LTL = 'convertLTL'

AutomataOperations = [OP_FSA_ACCEPTS, OP_FSA_IS_DETERMINISTIC, OP_FSA_AS_DFA, OP_FSA_ELIMINATE_EPSILON, OP_FSA_ALPHABET, OP_FSA_COMPLETE, OP_FSA_COMPLEMENT, OP_FSA_PRODUCT, OP_FSA_STRICT_PRODUCT, OP_FSA_PRODUCT_BUCHI, OP_FSA_STRICT_PRODUCT_BUCHI, OP_FSA_REACHABLE_STATES, OP_FSA_LANGUAGE_EMPTY, OP_FSA_LANGUAGE_EMPTY_BUCHI, OP_FSA_LANGUAGE_INCLUDED, OP_FSA_MINIMIZE, OP_FSA_MINIMIZE_BUCHI, OP_FSA_RELABEL, OP_FSA_AS_REGEX ]

RegExOperations = [OP_REGEX_CONVERT_REGEX, OP_REGEX_CONVERT_OMEGA_REGEX]

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
    OP_FSA_LANGUAGE_EMPTY + 'languageEmpty', 
    OP_FSA_LANGUAGE_EMPTY_BUCHI + 'languageEmptyBuchi', 
    OP_FSA_LANGUAGE_INCLUDED + 'languageIncluded', 
    OP_FSA_MINIMIZE + 'minimize', 
    OP_FSA_MINIMIZE_BUCHI + 'minimizeBuchi', 
    OP_FSA_RELABEL + 'relabel', 
    OP_FSA_AS_REGEX + 'asRegEx'
    OP_REGEX_CONVERT_REGEX + 'convertRegEx'
    OP_REGEX_CONVERT_OMEGA_REGEX + 'convertOmegaRegEx'
    OP_LTL_CONVERT_LTL + 'convertLTL'
]

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

