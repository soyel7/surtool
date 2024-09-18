#!/usr/bin/env bash

bash build.sh

docker save surgtoolloc_cqupt_0916 | gzip -c > surgtoolloc_cqupt_0916.tar.gz
