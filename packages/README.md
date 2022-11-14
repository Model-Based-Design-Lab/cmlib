# Analysis Tool Packages

## Prerequisites

Make sure that you have python 3.x.x installed, including pip.

## Install

Any of the packages can be installed as follows.
**Make sure to install the python-graph package from the github location, as described below, not from the standard Python repository.**

In a console (shell, command prompt or powershell), in this folder, type:

``` shell
$ python -m pip install "git+https://github.com/Shoobx/python-graph#egg=pkg&subdirectory=core"
$ cd markovchains
$ python -m pip install .
$ cd ../finitestateautomata
$ python -m pip install .
$ cd ../dataflow
$ python -m pip install .
```

Note that, depending on the python installation, you may need to have administrator / sudo rights to install the package. If the package is installed successfully, each analysis tool can be run as described in the following sections.


## finitestateautomata
>The finitestateautomata tool is capable of processing Finite State Automata (FSA), Regular Expressions (RegEx) and Linear Temporal Logic (LTL) models. The command describes which operation will be executed for the provided model. The commands that can be executed for each of the model types are listed in the tables below. For more information about the finitestateautomata tool and the required flags, run the command ```finitestateautomata -h``` in a terminal. Consider the example of running a command on a FSA model:
>```
>$ finitestateautomata model [-h] [-sa SECONDARYAUTOMATON] [-op OPERATION] [-oa OUTPUTAUTOMATON] [-re REGULAREXPRESSION] [-ltl LTLFORMULA] [-w INPUTTRACE]
>```
>Here, ```model``` can have the extentions ```.fsa```, ```.regex``` or ```.ltl``` for a FSA, RegEx or LTL model respectively. The model is a description in a plain text format. Epsilons are notated with the ```#``` symbol. Examples of these models can be found under ```\packages\finitestateautomata\finitestateautomata\tests\models\```. The table below describes the commands for the FSA models, together with the required flags. The same information is presented when running the ```-h``` flag.

### FSA

| Command                | Description                                                                      | required flags |
|------------------------|----------------------------------------------------------------------------------|----------------|
|```accepts```           | checks if provided word is accepted, requires input word                         |```-op, -w```   |
|```isDeterministic```   | check if automaton is deterministic                                              |```-op```       |
|```asDFA```             | convert FSA to DFA                                                               |```-op```       |
|```eliminateEpsilon```  | eliminate epsilon transitions from FSA                                           |```-op```       |
|```alphabet```          | determine the alphabet of the FSA                                                |```-op```       |
|```complete```          | make the FSA complete                                                            |```-op```       |
|```complement```        | determine the complement FSA                                                     |```-op```       |
|```product```           | determine product with secondary automaton                                       |```-op, -sa```  |
|```strictProduct```     | determine strict product with secondary automaton                                |```-op, -sa```  |
|```productBuchi```      | determine product of a Buchi automaton with secondary Buchi automaton            |```-op, -sa```  |
|```strictProductBuchi```| determine the strict product of a Buchi automaton with secondary Buchi automaton |```-op, -sa```  |
|```reachableStates```   | determine the reachable states of the automaton                                  |```-op```       |
|```languageEmpty```     | check if the language of the automaton is empty                                  |```-op```       |
|```languageEmptyBuchi```| check if the language of the Buchi automaton is empty                            |```-op```       |
|```languageIncluded```  | check if the language is included in the language of the secondary automaton     |```-op, -sa```  |
|```minimize```          | minimize the automaton                                                           |```-op```       |
|```minimizeBuchi```     | minimize the Buchi automaton                                                     |```-op```       |
|```relabel```           | relabel the states of the automaton                                              |```-op```       |
|```asRegEx```           | convert the automaton to a regular expression                                    |```-op```       |

### Regular Expressions
Like with the FSA models, the finitestateautomata tool can perform commands on RegEx models as follows. Here, the model has the extension ```model.regex``` and contains the description of the regular expression in plain text form. The operation converts the RegEx to a FSA.

| Command                | Description                                               | required flags |
|------------------------|-----------------------------------------------------------|----------------|
|```convertRegEx```      | convert the regular expresson to an FSA                   |```-op```       |
|```convertOmegaRegEx``` | convert the omega-regular expression to a Buchi automaton |```-op```       |

### Linear Temporal Logic
Lastly, the LTL model has the extension ```model.ltl``` and follows the same structure as the FSA and RegEx commands.


| Command         | Description                             | required flags |
|-----------------|-----------------------------------------|----------------|
|```convertLTL``` | convert the LTL formula to an automaton |```-op```       |


## Markovchains
> The markovchain tool can be categorized in two parts, the analytic and simulation operations. To get more information about the flags used in the markovchain tool, run the command ```markovchain -h``` in the terminal. Running ```markovchain -oh``` provides a list of all commands that can be used to analyze and simulate the markovchain models. The ```model``` represents the markovchain model which has the extention ```.dtmc```. Examples of markovchain models can be found in the ```\packages\markovchains\markovchains\test\models\``` directory.
>```
>$ markovchain model [-op OPERATION] [-ns NUMBEROFSTEPS] [-s TARGETSTATE] [-ss TARGETSTATESET] [-r STATEREWARDSET] [-sa STATESTARTINGSET] [-c CONDITIONS] [-sd SEED]
>```
> The simulation operations require stop conditions, which is a list of parameters containing the following fields:
> ```
> [confidence,abError,reError,numberOfSteps,numberOfPaths,timeInSeconds]
> ```
> ```confidence``` describes the confidence level, which has a range <0, 1>. ```abError``` is the absolute error and the ```reError``` is the relative error. When the simulation reaches these values, it will stop. The ```numberOfSteps``` sets the maximum number of steps the simulation will perform. Here, a step represents one transition from one state to the next state. ```numberOfPaths``` represents the maximum number of cycles before the simulation stops. One cycle represents the path (sequence of states) from a recurrent state to itself. Finally, ```timeInSeconds``` sets the maximum number of seconds the simulation can run. 

### Analytic commands
| Command                | Description                                               | required flags |
|------------------------|-----------------------------------------------------------|----------------|
|```liststates```|List all states of the markov chain|```-op```|
|```listrecurrentstates```|List all recurrent states of the markov chain|```-op```|
|```listtransientstates```|List all transient states of the markov chain|```-op```|
|```communicatingstates```|Provides list of communicating state sets|```-op```|
|```classifytransientrecurrent```|Lists the transient and recurrent states|```-op```|
|```hittingprobability```|Provides the hitting probability for a specified state|```-op, -s```|
|```hittingprobabilityset```|Provides the hitting probability for a specified state set|```-op, ss```|
|```rewardtillhit```|Expected reward until hitting specified state|```-op, -s```|
|```rewardtillhitset```|Expected reward until hitting specified state set|```-op, -ss```|
|```periodicity```|Lists aperiodic and periodic state states|```-op```|
|```mctype```|Provides type of markov chain: (non-)ergodic (non-)unichain|```-op```|
|```transient```|Transient analysis for specified number of steps|```-op, -ns```|
|```transientRewards```|Transient analysis of reward after specified number of steps|```-op, -ns```|
|```transientMatrix```|Transient matrix for specified number of steps|```-op, -ns```|
|```limitingMatrix```|Provides limiting Matrix|```-op```|
|```limitingDistribution```|Provides limiting Distribution|```-op```|
|```longRunReward```|Long-run expected average reward|```-op```|
|```executiongraph```|Prints execution graphs xml file for specified number of steps|```-op, -ns```|

### Simulation commands
The table below describes the simulation based operations which can be performed on the markovchain models. The flag in between brackets are optional. 
| Command                | Description                                               | required flags |
|------------------------|-----------------------------------------------------------|----------------|
|```markovtrace```|Provides simulation trace through markov chain|```-op, -ns, (-sd)```|
|```longrunexpectedaveragereward```|Long run expected average reward through simulation|```-op, -c, (-sd), (-s)```|
|```cezarolimitdistribution```|Cezarolimit distribution through simulation|```-op, -c, (-sd), (-s)```|
|```estimationexpectedreward```|Estimation of exected reward by simulation|```-op, -c, -ns, (-sd)```|
|```estimationdistribution```|Estimation of distribution by simulation after specified number of steps|```-op, -c, -ns, (-sd)```|
|```estimationhittingstate```|Estimation of hitting state probabilites by simulation|```-op, -c, -s, (-sd), (-sa)```|
|```estimationhittingreward```|Estimation of cumulative reward hitting state by simulation|```-op, -c, -s, (-sd), (-sa)```|
|```estimationhittingstateset```|Estimation of hitting state set probabilites by simulation|```-op, -c, -ss, (-sd), (-sa)```|
|```estimationhittingrewardset```|Estimation of cumulative reward hitting state set probabilites by simulation|```-op, -c, -ss, (-sd), (-sa)```|

Running a simulation with the stop conditions (-c flag) is best explained through an example. Consider the following markovchain $Model$. 
```
markov chain Model {
        S1 [p: 1; r: 1] -- 1/2 -> S2
        S1 -- 1/2 -> S6
        S2 [p: 0; r: 1] -- 2/3 -> S2
        S2 -- 1/3 -> S3
        S3 [p: 0; r: 1] -- 1/5 -> S4
        S3 -- 2/5 -> S5
        S3 -- 2/5 -> S6
        S4 [p: 0; r: 1] -- 1 -> S5
        S5 [p: 0; r: 1] -- 1 -> S4
        S6 [p: 0; r: 1] -- 1/4 -> S7
        S6 -- 3/4 -> S8
        S7 [p: 0; r: 1] -- 1/2 -> S7
        S7 -- 1/2 -> S6
        S8 [p: 0; r: 1] -- 1 -> S6
}
```
The example below shows a few operations which demonstrate the different behaviour when using the fields in the stop condition settings. In example 1., every path has a maximum length of 20 steps and every hitting probability is determined with 1000 paths (```[0.98,,,20,1000,1]```). The result show the hitting probability of the three starting states S1, S2 and S4 hitting state S7. In the second example, the number of paths is removed, causing the simulation to stop at a timeout of 1 second and evaluating only the hitting probability of state S7 from starting state S1. The reason for this is the unlimited number of paths for the evaluation of state S1 hitting state S7. The simulation then stops at the time requirement of 1 second. Finally, in the last example 3., also the maximum length of the path is removed, meaning that only one path is evaluated and no conclusion can be made of the hitting probability. This example tries to indicate that proper knowledge about the markovchain model and the simulation stop conditions is required to obtain meaningfull simulation results.
```
1.
$ markovchains dtmc_example3.dtmc -op estimationhittingstate -c [0.98,,,20,1000,1] -s S7 -sa S1,S2,S4
f(S1, S7) = 0.6430      int:[0.6078, 0.6782]    abEr:0.0352     reEr:0.0580     #paths:1000     stop:Number of Paths
f(S2, S7) = 0.3690      int:[0.3335, 0.4045]    abEr:0.0355     reEr:0.1064     #paths:1000     stop:Number of Paths
f(S4, S7) = 0.0000      int:[0.0000, 0.0000]    abEr:0.0000     reEr:None       #paths:1000     stop:Number of Paths

2.
$ markovchains dtmc_example3.dtmc -op estimationhittingstate -c [0.98,,,20,,1] -s S7 -sa S1,S2,S4
Estimated hitting probabilities for S7 are:
f(S1, S7) = 0.6457      int:[0.6392, 0.6521]    abEr:0.0065     reEr:0.0101     #paths:29715    stop:Timeout

3.
$ markovchains dtmc_example3.dtmc -op estimationhittingstate -c [0.98,,,,,1] -s S7 -sa S1,S2,S4
Estimated hitting probabilities for S7 are:
f(S1, S7) = None        int:[None, None]        abEr:None       reEr:None       #paths:1        stop:Timeout
```


## Dataflow
> The dataflow tool handles both Dataflow models and Max-Plus Matrix (MPM) models. To analyze either type, the command below can be run in the terminal. The ```model``` can have the extentions ```.sdf``` or ```.mpm```, representing Dataflow and MPM models respectively.
>```
>$ dataflow model [-h] [-op OPERATION] [-p PERIOD] [-is INITIALSTATE] [-it INPUTTRACE] [-ma MATRICES] [-sq SEQUENCES] [-pa PARAMETER] [-ni NUMBEROFITERATIONS] [-og OUTPUTGRAPH]
>```

### Dataflow models
| Command                | Description                                               | required flags |
|------------------------|-----------------------------------------------------------|----------------|
|```inputlabelssdf```|determines inputs of the graph|```-op```|
|```statelabelssdf```|determine the labels of the elements of the state vector|```-op```|
|```throughput```|computes throughput|```-op```|
|```repetitionvector```|computes repetition vector|```-op```|
|```latency```|requires period and optional initial state|```-op, -p```|
|```deadlock```|checks graph for deadlock|```-op```|
|```generalizedlatency```|compute generalized latency, requires period|```-op, -p```|
|```statematrix```|compute the state matrix of the graph|```-op```|
|```statespacerepresentation```|compute state space representation of the graph|```-op```|
|```statespacematrices```|compute state space representation of the graph as a new model|```-op```|
|```ganttchart```|make a Gantt chart of the graph as an XML file for the sdf3trace2svg tool|```-op, -ni```|
|```ganttchart-zero-based```|make a Gantt chart of the graph as an XML file for the sdf3trace2svg tool assuming actor firings cannot start before time 0|```-op```|
|```converttosinglerate```|convert to a single rate graph|```-op```|

### Max-Plus Matrix models
The flags within brackets are optional.
| Command                | Description                                               | required flags |
|------------------------|-----------------------------------------------------------|----------------|
|```eigenvalue```|compute the largest eigenvalue of a matrix|```-op```|
|```eventsequences```|list event sequences defined in the model|```-op```|
|```vectorsequences```|list vector sequences defined in the model|```-op```|
|```inputlabelsmpm```|determine the inputs of the model|```-op```|
|```matrices```|list matrices defined in the model|```-op```|
|```eigenvectors```|compute the eigenvectors of a matrix|```-op```|
|```precedencegraph```|compute the precedence graph of a matrix|```-op```|
|```precedencegraphgraphviz```|compute the precedence graph of a matrix as a Graphviz model|```-op```|
|```starclosure```|compute the star closure of a matrix|```-op```|
|```multiply```|multiply matrices and / or vector sequence; requires matrices possible a vector sequence|```-op, -ma```|
|```multiplytransform```|multiply matrices and / or vector sequence and make a new model with the result, requires matrices possible a vector sequence|```-op, -ma```|
|```vectortrace```|compute a vector trace for a state matrix or a set of state-space matrices|```-op, (-ni), (-is), (-sq)```|
|```vectortracetransform```|compute a vector trace for a state matrix or a set of state-space matrices as a new max-plus model|```-op, (-ni), (-is), (-sq)```|
|```vectortracexml```|compute a vector trace for a state matrix or a set of state-space matrices as an XML file for the sdf3trace2svg tool|```-op, (-ni), (-is), (-sq)```|
|```convolution```|compute the convolution of a series of event sequences|```-op, -sq```|
|```convolutiontransform```|compute the convolution of a series of event sequences as a new model|```-op, -sq```|
|```maxsequences```|compute the maximum of a series of event sequences|```-op, -sq```|
|```maxsequencestransform```|compute the maximum of a series of event sequences as a new model|```-op, -sq```|
|```delaysequence```|delay an event sequence by a number of event samples; requires sequence and parameter|```-op, -pa, -sq```|
|```scalesequence```|scale an event sequence, i.e., add a costant to all event time stamps; requires sequence and parameter|```-op, -pa, -sq````|
