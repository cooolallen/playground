agent0="${1}::null"
agent1="${2}::null"
agent2="${3}::null"
agent3="${4}::null"
nums_times=$5
# --record_json_dir=./battle

python battle/run_battle.py --agents=$agent0,$agent1,$agent2,$agent3 --num_times=$nums_times --config=PommeTeamCompetition-v0