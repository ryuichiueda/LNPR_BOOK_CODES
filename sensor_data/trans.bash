while read a b c ; do
	d=$(date -d @$a "+%Y%m%d %H%M%S")
	n=$(echo $c | awk '{print int(1000*($1+0.0005))}')
	echo $d $b $n
done
