#! /bin/bash

git fetch devremote
git merge devremote/master --allow-unrelated-histories
git push
