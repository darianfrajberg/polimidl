#!/bin/sh

if [ $# -ne 1 ]; then
    echo "Error! Missing argument";
    exit 1;
fi

for file in ./out/*; do
    if [ -f "$file" ] && [ -x "$file" ]; then
        echo ".............................."
        echo "Executing $(basename "${file}")..."
        ./"$file"
        echo ".............................."
    fi
done
