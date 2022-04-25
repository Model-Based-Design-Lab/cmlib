echo
echo "::: Markovtrace functions :::"
markovchains example1.dtmc -op markovtrace -ns 19
markovchains example1.dtmc -op markovtrace -ns 19 -sd 1
echo
echo
echo
echo "::: Longrunexpectedaveragereward functions :::" 
# Possible values for stop conditions
markovchains example1.dtmc -op longrunexpectedaveragereward -c [0.95,,,1000,,] # nr_of_steps
markovchains example1.dtmc -op longrunexpectedaveragereward -c [0.95,0.5,,1000,,] # abError
markovchains example1.dtmc -op longrunexpectedaveragereward -c [0.95,,0.5,1000,,] # reError
markovchains example1.dtmc -op longrunexpectedaveragereward -c [0.95,,,1000,,0.01] # time
# Possible flags
markovchains example1.dtmc -op longrunexpectedaveragereward -c [0.95,,,1000,,] -sd 1 # Seed
markovchains example1.dtmc -op longrunexpectedaveragereward -c [0.95,,,1000,,] -sd 1 -s C # Seed + recurrent state
echo
echo
echo
echo "::: Cezarolimitdistribution functions :::" 
# Possible values for stop conditions
markovchains example1.dtmc -op cezarolimitdistribution -c [0.95,,,1000,,] # nr_of_steps
markovchains example1.dtmc -op cezarolimitdistribution -c [0.95,0.5,,1000,,] # abError
markovchains example1.dtmc -op cezarolimitdistribution -c [0.95,,0.5,1000,,] # reError
markovchains example1.dtmc -op cezarolimitdistribution -c [0.95,,,1000,,0.01] # time
# Possible flags
markovchains example1.dtmc -op cezarolimitdistribution -c [0.95,,,1000,,] -sd 1 # Seed
markovchains example1.dtmc -op cezarolimitdistribution -c [0.95,,,1000,,] -sd 1 -s C # Seed + recurrent state
echo
echo
echo
echo
echo "::: estimationexpectedreward functions :::" 
# Possible values for stop conditions
markovchains example1.dtmc -op estimationexpectedreward -c [0.95,,,,100,] -ns 10 # nr_of_paths
markovchains example1.dtmc -op estimationexpectedreward -c [0.95,0.5,,,100,] -ns 10 # abError
markovchains example1.dtmc -op estimationexpectedreward -c [0.95,,0.5,,100,] -ns 10 # reError
markovchains example1.dtmc -op estimationexpectedreward -c [0.95,,,,100,0.001] -ns 10 # time
# Possible flags
markovchains example1.dtmc -op estimationexpectedreward -c [0.95,,,,100,] -ns 10 -sd 1 # seed
echo
echo
echo
echo
echo "::: estimationdistribution functions :::" 
# Possible values for stop conditions
markovchains example1.dtmc -op estimationdistribution -c [0.95,,,,100,] -ns 10 # nr_of_paths
markovchains example1.dtmc -op estimationdistribution -c [0.95,0.5,,,100,] -ns 10 # abError
markovchains example1.dtmc -op estimationdistribution -c [0.95,,0.5,,100,] -ns 10 # reError
markovchains example1.dtmc -op estimationdistribution -c [0.95,,,,100,0.001] -ns 10 # time
# Possible flags
markovchains example1.dtmc -op estimationdistribution -c [0.95,,,,100,] -ns 10 -sd 1 # seed
echo
echo
echo
echo
echo "::: estimationhittingstate functions :::" 
# Possible values for stop conditions
markovchains example3.dtmc -op estimationhittingstate -c [0.95,,,100,100,] -s S6 # nr_of_paths
markovchains example3.dtmc -op estimationhittingstate -c [0.95,0.5,,100,100,] -s S6 # abError
markovchains example3.dtmc -op estimationhittingstate -c [0.95,,0.5,100,100,] -s S6 # reError
markovchains example3.dtmc -op estimationhittingstate -c [0.95,,,100,100,0.05] -s S6 # time
# Possible flags
markovchains example3.dtmc -op estimationhittingstate -c [0.95,,,100,100,] -s S6 -sd 1 # seed
markovchains example3.dtmc -op estimationhittingstate -c [0.95,,,100,100,] -s S6 -sd 1 -sa S1,S2,S3 # seed + state selection
echo
echo
echo
echo
echo "::: estimationhittingreward functions :::" 
# Possible values for stop conditions
markovchains example3.dtmc -op estimationhittingreward -c [0.95,,,100,100,] -s S6 # nr_of_paths
markovchains example3.dtmc -op estimationhittingreward -c [0.95,0.5,,100,100,] -s S6 # abError
markovchains example3.dtmc -op estimationhittingreward -c [0.95,,0.5,100,100,] -s S6 # reError
markovchains example3.dtmc -op estimationhittingreward -c [0.95,,,100,100,0.01] -s S6 # time
# Possible flags
markovchains example3.dtmc -op estimationhittingreward -c [0.95,,,100,100,] -s S6 -sd 1 # seed
markovchains example3.dtmc -op estimationhittingreward -c [0.95,,,100,100,] -s S6 -sd 1 -sa S1,S2,S3 # seed + state selection
echo
echo
echo
echo
echo "::: estimationhittingstateset functions :::" 
# Possible values for stop conditions
markovchains example3.dtmc -op estimationhittingstateset -c [0.95,,,100,100,] -ss S6,S7 # nr_of_paths
markovchains example3.dtmc -op estimationhittingstateset -c [0.95,0.5,,100,100,] -ss S6,S7 # abError
markovchains example3.dtmc -op estimationhittingstateset -c [0.95,,0.5,100,100,] -ss S6,S7 # reError
markovchains example3.dtmc -op estimationhittingstateset -c [0.95,,,100,100,0.05] -ss S6,S7 # time
# Possible flags
markovchains example3.dtmc -op estimationhittingstateset -c [0.95,,,100,100,] -ss S6,S7 -sd 1 # seed
markovchains example3.dtmc -op estimationhittingstateset -c [0.95,,,100,100,] -ss S6,S7 -sd 1 -sa S1,S2,S3 # seed + state selection
echo
echo
echo
echo
echo "::: estimationhittingrewardset functions :::" 
# Possible values for stop conditions
markovchains example3.dtmc -op estimationhittingrewardset -c [0.95,,,100,100,] -ss S6,S7 # nr_of_paths
markovchains example3.dtmc -op estimationhittingrewardset -c [0.95,0.5,,100,100,] -ss S6,S7 # abError
markovchains example3.dtmc -op estimationhittingrewardset -c [0.95,,0.5,100,100,] -ss S6,S7 # reError
markovchains example3.dtmc -op estimationhittingrewardset -c [0.95,,,100,100,0.01] -ss S6,S7 # time
# Possible flags
markovchains example3.dtmc -op estimationhittingrewardset -c [0.95,,,100,100,] -ss S6,S7 -sd 1 # seed
markovchains example3.dtmc -op estimationhittingrewardset -c [0.95,,,100,100,] -ss S6,S7 -sd 1 -sa S1,S2,S3 # seed + state selection
