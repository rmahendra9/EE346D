#!/bin/bash

# Check if the script is run as root
# ref: https://askubuntu.com/a/30157/8698
if ! [ $(id -u) = 0 ]; then
   echo "The script need to be run as root." >&2
   exit 1
fi

# Check if an IP address is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <Subnet CIDR>"
    exit 1
fi

subnet_cidr=$1
firewall-cmd --zone=trusted --add-source="$subnet_cidr"