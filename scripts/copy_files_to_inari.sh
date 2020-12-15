#!/bin/bash

base_dir=${PWD}

set -x
scp -rp "$base_dir"/data salamid1@inari.comnet.aalto.fi:/home/salamid1/graph-convolution/codes
scp -rp "$base_dir"/models salamid1@inari.comnet.aalto.fi:/home/salamid1/graph-convolution/codes
scp "$base_dir"/* salamid1@inari.comnet.aalto.fi:/home/salamid1/graph-convolution/codes
set +x