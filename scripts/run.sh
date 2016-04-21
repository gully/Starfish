#!/bin/bash

cp ../../../s0_o0phi.json .

mkdir plots

time star.py --sample=ThetaPhi --samples=10

time star.py --sample=ThetaPhi --samples=1000

chain.py --files mc.hdf5 --chain

open walkers.png

chain_run.py --chain

open s0_o0/walkers.png

echo "-------------------------------------"
echo "Manually input values into the config.yaml file."
echo "-------------------------------------"
read  -p "Waiting while you do this.  Type any key to continue."


echo "-------------------------------------"
echo "The end for now."
echo "-------------------------------------"