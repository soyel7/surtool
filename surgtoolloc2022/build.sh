#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t surgtoolloc_cqupt_0916 "$SCRIPTPATH"
