#!/bin/bash

NUMTIMES=1
DELAYMIN=0

if [ "$#" -eq 2 -o "$#" -eq 4 ]; then
	FILE_SIZE=${2}
	if [ "$#" -eq 4 ]; then
		NUMTIMES=${3}
		DELAYMIN=${4}
	fi
else
	echo "Please use: file_name file_size_in_kB"
	echo "Or: file_name file_size_in_kB number_of_times delay_in_min"
	exit 1
fi

HOST_RX=root@192.168.1.2
HOST_TX=root@192.168.1.1
PACKETS_NUM=200

for ((i=0;i<$NUMTIMES;i++))
do
	FILE_NAME=$(date +D=%F_T=%H-%M-%S)--${1}'.dat'
	ssh $HOST_RX ./RX /tmp/$FILE_NAME $FILE_SIZE &
	RX_PID=$!
	
	while true
	do
		if ! kill -0 $RX_PID > /dev/null 2>&1; then
					scp $HOST_RX:/tmp/$FILE_NAME $FILE_NAME
					ssh $HOST_RX rm -f /tmp/$FILE_NAME
			break
		fi
		ssh $HOST_TX ./TX $PACKETS_NUM
	done
	
	if [ $i -ne $(($NUMTIMES-1)) ]; then
		echo "Sleeping $DELAYMIN min..."
		sleep $((DELAYMIN))m
	fi
done