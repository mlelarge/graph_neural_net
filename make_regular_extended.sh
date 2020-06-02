#/bin/sh

for n in $(seq 0 0.005 0.2)
do
	for t in {0..9}
	do
		echo "Repetition $t with noise $n"
		python commander.py with cpu=False name=RegularExtended data.generative_model=Regular data.noise=$n data.vertex_proba=0.9
	done
done
