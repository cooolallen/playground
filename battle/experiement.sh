agent0="${1}::null"
agent1="${2}::null"
agent2="${3}::null"
agent3="${4}::null"
nums_times=$5
iter_num=$6

dir_name="output_${1}_${2}"

mkdir -p ${dir_name}

for i in $(seq 1 $iter_num)
do
    python battle/run_battle.py --agents=$agent0,$agent1,$agent2,$agent3 --num_times=$nums_times --config=PommeTeamCompetition-v0 &> ${dir_name}/output_${i}.txt &
done