#!/bin/bash

# ref: https://askubuntu.com/a/30157/8698
if ! [ $(id -u) = 0 ]; then
   echo "The script need to be run as root." >&2
   exit 1
fi

if [ $SUDO_USER ]; then
    real_user=$SUDO_USER
else
    real_user=$(whoami)
fi

#ref: https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb
apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.10 python3.10-venv python3.10-dev -y
rm /usr/bin/python3
ln -s python3.10 /usr/bin/python3
sudo -u $real_user python3 --version
sudo -u $real_user curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
sudo -u $real_user python3 -m pip --version
sudo -u $real_user python3 -m venv .venv
