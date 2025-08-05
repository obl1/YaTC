#!/bin/bash

# Define options: name, URL, and nclasses
options=(
    "AES128 https://drive.google.com/uc?id=1IMCpgqHOE0HG8MHqxksqpQL_5dYMIg56"
    "AES256 https://drive.google.com/uc?id=1awcAb2IJDxDDV_-RrBGtA1avP98jzvhH"
    "CHACHA https://drive.google.com/uc?id=1M9N_Q0J2robRsIN0neUizWKrORANFUJp"
    "MIX https://drive.google.com/uc?id=1Q3G36FArdg9x88i_W8-cv9l8Uzl0yN03"
)

template="spectrum_%s_YATC_D12_nogetpocket"

for option in "${options[@]}"; do
    read -r cipher url nclasses <<< "$option"
    name=$(printf "$template" "$cipher")
    
    # Print the parameters before running
    echo "Running generic_lambda_yatc.sh with:"
    echo "  Name:      $name"
    echo "  URL:       $url"
    echo "  NClasses:  $nclasses"
    echo
    
    ./generic_lambda_yatc.sh "$name" "$url" "$nclasses"
done
