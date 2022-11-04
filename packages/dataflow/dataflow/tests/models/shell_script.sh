#!/bin/sh

runSDFs()
{
    echo "--- processing [$1] ---"
    for f in sdf_channel_decoder.sdf sdf_ConveyorBelt.sdf sdf_deadlock.sdf sdf_eggs.sdf sdf_multiple_outputs.sdf sdf_prod_cons.sdf sdf_railroad.sdf; do
        echo "- file [$f] -"
        dataflow $f -op $1
        echo
    done
    echo
}
runMPMs()
{
    echo "--- processing [$1] ---"
    for m in mpm_channel_decoder.mpm mpm_eggs.mpm mpm_EventSequences.mpm mpm_MatrixExample.mpm mpm_MatrixExample2.mpm mpm_prod_cons.mpm; do
        echo "- file [$m] -"
        dataflow $m -op $1
        echo
    done
    echo
}

# runSDFs inputlabelssdf
# runSDFs statelabelssdf
# runSDFs throughput
# runSDFs repetitionvector
# runSDFs "latency -p 100"
# runSDFs deadlock
# runSDFs "generalizedlatency -p 10"
# runSDFs statematrix
# runSDFs statespacerepresentation
# runSDFs statespacematrices
# runSDFs "ganttchart -ni 4"
# runSDFs "ganttchart-zero-based -ni 4"
# runSDFs converttosinglerate

runMPMs eigenvalue
runMPMs eventsequences
runMPMs vectorsequences
runMPMs inputlabelsmpm
runMPMs matrices
runMPMs eigenvectors
runMPMs precedencegraph
runMPMs precedencegraphgraphviz
runMPMs starclosure
runMPMs "multiply -ma A,A"
runMPMs "multiplytransform -ma A,A"
runMPMs "vectortrace -it [[0,8,16,32],[1,9,17,33]]"
runMPMs "vectortracetransform -it [[0,8,16,32],[1,9,17,33]]"
runMPMs "vectortracexml -it [[0,8,16,32],[1,9,17,33]]"
dataflow mpm_EventSequences.mpm -op convolution -sq h,x
dataflow mpm_MatrixExample2.mpm -op convolution -sq delta,delta
dataflow mpm_EventSequences.mpm -op convolutiontransform -sq h,x
dataflow mpm_MatrixExample2.mpm -op convolutiontransform -sq delta,delta
dataflow mpm_EventSequences.mpm -op maxsequences -sq h,x
dataflow mpm_EventSequences.mpm -op maxsequencestransform -sq h,x
dataflow mpm_EventSequences.mpm -op delaysequence -sq h -pa 2
dataflow mpm_MatrixExample2.mpm -op delaysequence -sq delta -pa 6
dataflow mpm_EventSequences.mpm -op scalesequence -sq h -pa 2
