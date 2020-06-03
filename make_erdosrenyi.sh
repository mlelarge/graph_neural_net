#/bin/sh

for n in 0.0 0.005 0.01 0.015 0.02 0.025 0.03 0.04 0.05
do
	for t in {0..9}
	do
		echo "Repetition $t with noise $n"
		python commander.py with cpu=False name=ErdosRenyi data.generative_model=ErdosRenyi data.noise=$n
	done
done
