#!/bin/bash
# Usage: ./figures.sh
# Makes plots of algs (add,div) x dist (zipf) x noise (0,0.15) x datasets (random, odors, mnist)


exe=./fly_counting.py # version to run.
noise=0.15

# additive / Hebbian model.
$exe -d random -a add -t zipf 
$exe -d random -a add -t zipf -x $noise

$exe -d odors -a add -t zipf
$exe -d odors -a add -t zipf -x $noise

$exe -d mnist -a add -t zipf
$exe -d mnist -a add -t zipf -x $noise


# divisive / anti-Hebbian model.
$exe -d random -a div -t zipf 
$exe -d random -a div -t zipf -x $noise

$exe -d odors -a div -t zipf 
$exe -d odors -a div -t zipf -x $noise

$exe -d mnist -a div -t zipf 
$exe -d mnist -a div -t zipf -x $noise